import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..import_utils import is_bnb_4bit_available, is_bnb_available
from ..utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)
from .lora import (
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
)
import numpy as np

if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class TriAdaptLoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.AdaLora`].

    Args:
        reference_rank (`int`): The target average rank of incremental matrix.
        init_rank (`int`): The initial rank for each incremental matrix.
        init_warmup (`int`): The steps of initial fine-tuning warmup.
        incre_interval (`int`): The time internval between two budget allocations.
        top_k (`int`): Fixed rank growth threshold mode.
        orth_reg_weight (`float`): The coefficient of orthogonal regularization.
        weight_decay (`float`): Weight decay factor.
        rank_growth_model (`str`): Adaptive rank growth model.
        incre_rank_num (`int`): The size of each rank increase.
        total_step (`int`): The total training steps that should be specified before training.
        target_total_rank (`int`): Total rank budget size.
        rank_pattern (`list`): The allocated rank for each weight matrix by RankAllocator.
    """

    reference_rank: int = field(default=8, metadata={"help": "Target Lora matrix dimension."})
    init_rank: int = field(default=1, metadata={"help": "Intial Lora matrix dimension."})
    init_warmup: int = field(default=0, metadata={"help": "The steps of initial warmup."})
    incre_interval: int = field(default=1, metadata={"help": "Step interval of rank allocation."})
    top_k: int = field(default=1, metadata={"help": "Fixed rank growth threshold mode."})
    orth_reg_weight: float = field(default=0.5, metadata={"help": "The orthogonal regularization coefficient."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay factor."})
    rank_growth_model: str = field(default="linear", metadata={"help": "Adaptive rank growth model."})
    incre_rank_num: Optional[int] = field(default=None, metadata={"help": "The size of each rank increase."})
    total_step: Optional[int] = field(default=None, metadata={"help": "The total training steps."})
    target_total_rank: Optional[int] = field(default=None, metadata={"help": "Total rank budget size."})
    rank_pattern: Optional[dict] = field(default=None, metadata={"help": "The saved rank pattern."})

    def __post_init__(self):
        self.peft_type = PeftType.TRIADAPTLORA


class TriAdaptLoraModel(LoraModel):
    """
    Creates TriAdaptLoRA model from a pretrained transformers model. Paper:
    https://arxiv.org/pdf/2501.08008

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`TriAdaptLoraConfig`]): The configuration of the AdaLora model.

    Returns:
        `torch.nn.Module`: The AdaLora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import TriAdaptLoraModel, TriAdaptLoraConfig
        >>> config = TriAdaptLoraConfig(
                peft_type="TRIADAPTLORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
                lora_dropout=0.01,
            )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> model = TriAdaptLoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`TriAdaptLoraConfig`]): The configuration of the AdaLora model.
    """

    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_triadaptlora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "TriAdaptLoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        traininable_mode_counter = 0
        for config in self.peft_config.values():
            if not config.inference_mode:
                traininable_mode_counter += 1

        if traininable_mode_counter > 1:
            raise ValueError(
                "TriAdaptLoraModel supports only 1 trainable adapter. "
                "When using multiple adapters, set inference_mode to True for all adapters except the one you want to train."
            )

        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)
        else:
            self.trainable_adapter_name = adapter_name
            self.rankallocator = RankAllocator(self.model, self.peft_config[adapter_name], self.trainable_adapter_name)

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)

        if (loaded_in_8bit or loaded_in_4bit) and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "r": lora_config.init_rank,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(lora_config.target_modules, str):
                target_module_found = re.fullmatch(lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                bias = target.bias is not None
                if isinstance(target, LoraLayer):
                    target.update_layer(
                        adapter_name,
                        lora_config.init_rank,
                        lora_config.lora_alpha,
                        lora_config.lora_dropout,
                        lora_config.init_lora_weights,
                    )
                else:
                    if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                        kwargs.update(
                            {
                                "has_fp16_weights": target.state.has_fp16_weights,
                                "memory_efficient_backward": target.state.memory_efficient_backward,
                                "threshold": target.state.threshold,
                                "index": target.index,
                            }
                        )
                        new_module = TriAdaptLinear8bitLt(
                            adapter_name, target.in_features, target.out_features, bias=bias, **kwargs
                        )
                    elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
                        fourbit_kwargs = kwargs.copy()
                        fourbit_kwargs.update(
                            {
                                "compute_dtype": target.compute_dtype,
                                "compress_statistics": target.weight.compress_statistics,
                                "quant_type": target.weight.quant_type,
                            }
                        )
                        new_module = TriAdaptLinear4bit(
                            adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs
                        )
                    else:
                        if isinstance(target, torch.nn.Linear):
                            in_features, out_features = target.in_features, target.out_features
                            if kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                                    "Setting fan_in_fan_out to False."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
                        elif isinstance(target, Conv1D):
                            in_features, out_features = (
                                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                            )
                            if not kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                                    "Setting fan_in_fan_out to True."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                            )
                        new_module = TriAdaptLinear(adapter_name, in_features, out_features, bias=bias, **kwargs)

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, *args, **kwargs):
        outputs = self.model.forward(*args, **kwargs)
        orth_reg_weight = self.peft_config[self.trainable_adapter_name].orth_reg_weight
        if orth_reg_weight == 0:
            return outputs
        # Calculate the orthogonal regularization
        print("=====================================Enter the orthogonal regularization=====================================")
        if hasattr(outputs, "loss"):
            regu_loss = 0
            num_param = 0

            def compute_regu(para_cov):
                I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
                I.requires_grad = False
                return torch.norm(para_cov - I, p="fro")

            for n, layer in self.model.named_modules():
                if isinstance(layer, TriAdaptLoraLayer) and self.trainable_adapter_name in layer.lora_A.keys():
                    wA = torch.cat([a for a in layer.lora_A[self.trainable_adapter_name]], 0)
                    wB = torch.cat([b for b in layer.lora_B[self.trainable_adapter_name]], 1)
                    para_cov_A = wA @ wA.T
                    para_cov_B = wB.T @ wB

                    if regu_loss is None:
                        regu_loss = compute_regu(para_cov_A)
                    else:
                        regu_loss += compute_regu(para_cov_A)
                    regu_loss += compute_regu(para_cov_B)
                    num_param += 2

            if num_param > 0:
                regu_loss = regu_loss / num_param
            else:
                regu_loss = 0
            outputs.loss += orth_reg_weight * regu_loss
        return outputs

    def update_and_increase(self, global_step, optimizer):
        self.rankallocator.update_and_increase(self.model, global_step, optimizer)

    def set_total_step(self, total_step: int):
        self.rankallocator.set_total_step(total_step)

    def get_rank_pattern(self):
        # Return rank pattern
        return self.rankallocator.get_rank_pattern()

    def _unload_and_optionally_merge(self, merge=True):
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                else:
                    bias = target.bias is not None
                    if getattr(target, "is_target_conv_1d_layer", False):
                        new_module = Conv1D(target.out_features, target.in_features)
                    else:
                        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    print(f"start merging TriAdaptLoRA key: {key}")
                    target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    @staticmethod
    def _prepare_triadaptlora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        return peft_config


class TriAdaptLoraLayer(LoraLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__(in_features, out_features)
        self.lora_C_row = nn.ParameterDict({})
        self.lora_C_col = nn.ParameterDict({})
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        self.ranknum = {}
        self.score = {}
        self.preNormValue = {}

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:

            def lora_dropout_layer(x):
                return x

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(nn.ParameterDict({adapter_name: nn.ParameterList([])}))
            self.lora_C_row.update(nn.ParameterDict({adapter_name: nn.ParameterList([])}))
            self.lora_C_col.update(nn.ParameterDict({adapter_name: nn.ParameterList([])}))
            self.lora_B.update(nn.ParameterDict({adapter_name: nn.ParameterList([])}))
            self.W = loraW()
            self.gradMatrix_trace = 0
            # The current rank
            self.ranknum[adapter_name] = 0
            self.score[adapter_name] = 0.0
            self.preNormValue[adapter_name] = 0.0
            self.scaling[adapter_name] = lora_alpha if lora_alpha > 0 else float(r)
            print("=====================================Enter the new TriAdaptLoraLayer linear F divided by r method========================================")
            self.add_parameters(r, adapter_name)

        # The current rank
        self.scaling[adapter_name] = lora_alpha if lora_alpha > 0 else float(r)
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            print("==================================Enter the reset_lora_parameters method=====================================")
            # nn.init.normal_(self.lora_C_row[adapter_name][0], mean=0.0, std=0.02)
            # nn.init.normal_(self.lora_C_col[adapter_name][0], mean=0.0, std=0.02)
            nn.init.normal_(self.lora_A[adapter_name][0], mean=0.0, std=0.02)
            # nn.init.zeros_(self.lora_B[adapter_name][0])
            self.compute_score(adapter_name)

    def add_parameters(self, add_r, adapter_name):
        print("==================================Enter the add_parameters method=====================================")
        if len(self.lora_C_row[adapter_name]) > 0:
            ranknum = self.lora_C_row[adapter_name][-1].size(1)
        else:
            ranknum = 0
        size = ranknum + add_r
        c_row = nn.Parameter(self.weight.new_zeros((add_r, size)), requires_grad=True)
        c_col = nn.Parameter(self.weight.new_zeros((size, add_r)), requires_grad=True)
        a = nn.Parameter(self.weight.new_zeros((add_r, self.in_features)), requires_grad=True)
        b = nn.Parameter(self.weight.new_zeros((self.out_features, add_r)), requires_grad=True)
        # nn.init.normal_(c_row, mean=0.0, std=0.02)
        # nn.init.normal_(c_col, mean=0.0, std=0.02)
        # nn.init.normal_(a, mean=0.0, std=0.02)
        # nn.init.normal_(b, mean=0.0, std=0.02)
        self.lora_C_row[adapter_name].append(c_row)
        self.lora_C_col[adapter_name].append(c_col)
        self.lora_A[adapter_name].append(a)
        self.lora_B[adapter_name].append(b)
        # rank increase add_r
        self.ranknum[adapter_name] += add_r

    def compute_score(self, adapter_name):
        C_row = self.lora_C_row[adapter_name]
        C_col = self.lora_C_col[adapter_name]
        ranknum = C_row[-1].size(1)
        size = len(C_row)
        device = C_row[0].device
        dtype = C_row[0].dtype
        # Concatenate the lower triangular matrix
        c_row_matrix = torch.cat(
            [torch.cat(
                [C_row[i], torch.zeros(C_row[i].size(0), ranknum - C_row[i].size(1), device=device, dtype=dtype)],
                dim=1)
                for i in range(size)], dim=0)
        # Concatenate the upper triangular matrix
        c_col_matrix = torch.cat(
            [torch.cat(
                [C_col[i], torch.zeros(ranknum - C_col[i].size(0), C_col[i].size(1), device=device, dtype=dtype)],
                dim=0)
                for i in range(size)], dim=1)

        C_concat = c_row_matrix + c_col_matrix
        lora_C_norm = torch.norm(C_concat, p='fro')
        normalized_lora_C_norm = lora_C_norm / torch.tensor(size).float()
        self.score[adapter_name] = normalized_lora_C_norm - self.preNormValue[adapter_name]
        self.preNormValue[adapter_name] = normalized_lora_C_norm
        return self.score[adapter_name]

class loraW(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, C_row, C_col, B, scaling):
        ranknum = C_row[-1].size(1)
        size = len(C_row)
        device = C_row[0].device
        dtype = C_row[0].dtype
        # Concatenate the lower triangular matrix
        c_row_matrix = torch.cat(
            [torch.cat(
                [C_row[i], torch.zeros(C_row[i].size(0), ranknum - C_row[i].size(1), device=device, dtype=dtype)],
                dim=1)
             for i in range(size)], dim=0)
        # Concatenate the upper triangular matrix
        c_col_matrix = torch.cat(
            [torch.cat(
                [C_col[i], torch.zeros(ranknum - C_col[i].size(0), C_col[i].size(1), device=device, dtype=dtype)],
                dim=0)
             for i in range(size)], dim=1)
        C_concat = c_row_matrix + c_col_matrix
        A_concat = torch.cat([a for a in A], 0)
        B_concat = torch.cat([b for b in B], 1)

        return B_concat @ (C_concat @ A_concat) * scaling / (ranknum + 1e-5)

class TriAdaptLinear(nn.Linear, TriAdaptLoraLayer):
    # Triangular Adaptive Low-Rank Adaptation
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        TriAdaptLoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def get_rank(self, adapter_name):
        return self.ranknum[adapter_name]

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r[self.active_adapter] > 0:
            print("==============start TriAdaptLoRA merging==============")
            self.weight.data += T(
                self.W(self.lora_A[self.active_adapter], self.lora_C_row[self.active_adapter],
                       self.lora_C_col[self.active_adapter], self.lora_B[self.active_adapter],
                       self.scaling[self.active_adapter])
            )
            self.merged = True

    def unmerge(self):
        print("start TriAdaptLoRA unmerging")
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r[self.active_adapter] > 0:
            self.weight.data -= T(
                self.W(self.lora_A[self.active_adapter], self.lora_C_row[self.active_adapter],
                       self.lora_C_col[self.active_adapter], self.lora_B[self.active_adapter],
                       self.scaling[self.active_adapter])
            )
            self.merged = False

    def forward(self, x: torch.Tensor):
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            result += (
                self.lora_dropout[self.active_adapter](x)
                @ self.W(self.lora_A[self.active_adapter], self.lora_C_row[self.active_adapter],
                         self.lora_C_col[self.active_adapter], self.lora_B[self.active_adapter],
                         self.scaling[self.active_adapter]).T
            )
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        return result


if is_bnb_available():

    class TriAdaptLinear8bitLt(bnb.nn.Linear8bitLt, TriAdaptLoraLayer):
        # Low-rank matrix for SVD-based adaptation
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            TriAdaptLoraLayer.__init__(self, in_features=in_features, out_features=out_features)
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
                return result
            elif self.r[self.active_adapter] > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = (
                        self.lora_dropout[self.active_adapter](x) @
                        self.W(self.lora_A[self.active_adapter], self.lora_C_row[self.active_adapter],
                               self.lora_C_col[self.active_adapter], self.lora_B[self.active_adapter],
                               self.scaling[self.active_adapter]).T
                    ).to(expected_dtype)
                else:
                    output = (
                        self.lora_dropout[self.active_adapter](x) @
                        self.W(self.lora_A[self.active_adapter], self.lora_C_row[self.active_adapter],
                               self.lora_C_col[self.active_adapter], self.lora_B[self.active_adapter],
                               self.scaling[self.active_adapter]).T
                    )
                result = result + output
            return result


if is_bnb_4bit_available():

    class TriAdaptLinear4bit(bnb.nn.Linear4bit, TriAdaptLoraLayer):
        # Low-rank matrix for SVD-based adaptation
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            bnb.nn.Linear4bit.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                compute_dtype=kwargs.get("compute_dtype", torch.float32),
                compress_statistics=kwargs.get("compress_statistics", True),
                quant_type=kwargs.get("quant_type", "nf4"),
            )
            TriAdaptLoraLayer.__init__(self, in_features=in_features, out_features=out_features)
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
                return result
            elif self.r[self.active_adapter] > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = (
                        self.lora_dropout[self.active_adapter](x) @
                        self.W(self.lora_A[self.active_adapter], self.lora_C_row[self.active_adapter],
                               self.lora_C_col[self.active_adapter], self.lora_B[self.active_adapter],
                               self.scaling[self.active_adapter]).T
                    ).to(expected_dtype)
                else:
                    output = (
                        self.lora_dropout[self.active_adapter](x) @
                        self.W(self.lora_A[self.active_adapter], self.lora_C_row[self.active_adapter],
                               self.lora_C_col[self.active_adapter], self.lora_B[self.active_adapter],
                               self.scaling[self.active_adapter]).T
                    )
                result = result + output
            return result


class RankAllocator(object):
    """
    The RankAllocator for TriAdaptLoraModel. Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        config ([`TriAdaptLoraConfig`]): The configuration of the AdaLora model.
        model: the model that we apply AdaLoRA to.

    """

    def __init__(self, model, peft_config, adapter_name):
        self.peft_config = peft_config
        self.adapter_name = adapter_name
        self.reference_rank = peft_config.reference_rank
        self.target_total_rank = peft_config.target_total_rank
        self.rank_growth_model = peft_config.rank_growth_model
        self.init_warmup = peft_config.init_warmup
        self.incre_interval = peft_config.incre_interval
        self.top_k = peft_config.top_k
        if peft_config.incre_rank_num:
            self.incre_rank_num = peft_config.incre_rank_num
        else:
            rank_dic = {2:1, 4:2, 6:3, 8:4, 10:5, 12:6, 14:7, 16:8, 18:9, 20:10, 22:11, 24:12, 26:13, 28:14, 30:15,
                        32:16, 34:17, 36:18, 38:19, 40:20, 42:21, 44:22, 46:23, 48:24, 50:25, 52:26, 54:27, 56:28,
                        58:29, 60:30, 62:31, 64:32, 66:33, 68:34, 70:35, 72:36, 74:37, 76:38, 78:39, 80:40, 82:41,
                        84:42, 86:43, 88:44, 90:45, 92:46, 94:47, 96:48, 98:49, 100:50, 128:64}
            self.incre_rank_num = rank_dic[self.reference_rank]
        self.total_step = peft_config.total_step
        self.model = model
        self.weight_decay = peft_config.weight_decay
        self.ipt = {}
        self.cat_ipt = {}
        self.rank_pattern = {}
        self._set_budget_scheduler()
        self.total_rank = self.initial_total_rank

    def set_total_step(self, total_step:int):
        self.total_step = total_step

    def get_rank_pattern(self):
        # Return rank pattern
        return self.rank_pattern

    def _set_budget_scheduler(self):
        # Prepare the budget scheduler
        self.name_set = set()
        self.initial_total_rank = 0
        self.shape_dict = {}
        for n, layer in self.model.named_modules():
            if isinstance(layer, TriAdaptLoraLayer) and self.adapter_name in layer.lora_A.keys():
                self.name_set.add(n)
                self.initial_total_rank += layer.lora_A[self.adapter_name][0].size(0)
                self.shape_dict[n + '.lora_A'] = layer.lora_A[self.adapter_name][0].shape
                self.shape_dict[n + '.lora_B'] = layer.lora_B[self.adapter_name][0].shape

        self.name_set = list(sorted(self.name_set))
        if self.target_total_rank is None:
            self.target_total_rank = self.reference_rank * len(self.name_set)

    def compute_threshold(self, global_step:int, all_is:list):
        """
        The threshold value for rank increase calculation.

        Args:
        - global_step: Current training step.
        - all_is: Dictionary of importance scores for parameter matrices.

        Returns:
        - increase_threshold: Threshold value for rank increase.
        """
        init_warmup = self.init_warmup
        total_step = self.total_step
        self.global_step = global_step
        num_increase_matrices = 0
        # Calculate the coefficient of the global budget.
        mul_coeff = (global_step-init_warmup)/(total_step-init_warmup)
        print("self.target_total_rank :", self.target_total_rank)
        print("self.total_rank :", self.total_rank)
        print("mul_coeff :", mul_coeff)
        # Calculate the number of parameter matrices that require rank increase.
        if self.rank_growth_model == "linear":
            # Linear growth.
            print("Performing linear rank growth.")
            num_increase_matrices = int((self.target_total_rank - self.total_rank) * (mul_coeff))
        elif self.rank_growth_model == "non-linear":
            # Exponential growth.
            print("Performing non-linear rank growth.")
            num_increase_matrices = int(np.exp(mul_coeff * np.log(self.target_total_rank - self.total_rank)))
        else:
            print("Invalid rank_growth_model value.")
            return "Error: Invalid model"
        print("num_increase_matrices :", num_increase_matrices)
        if num_increase_matrices <= 0:
            num_increase_matrices = 1
        # Convert the importance scores into a tensor and calculate the threshold value for rank increase.
        importance_scores = torch.tensor(all_is)
        # print("importance_scores :", importance_scores)
        num_increase_matrices = min(num_increase_matrices, len(importance_scores))
        print("torch.topk(importance_scores, num_increase_matrices) :", torch.topk(importance_scores, num_increase_matrices))
        increase_threshold = torch.topk(importance_scores, num_increase_matrices)[0][-1].item()

        return increase_threshold

    def increase_to_target_rank(self, model, optimizer, global_step):
        is_dict = {}
        all_is = []
        # Calculate the importance score for each sub matrix
        for n, layer in model.named_modules():
            if isinstance(layer, TriAdaptLoraLayer) and self.adapter_name in layer.lora_A.keys():
                ipt_score = layer.compute_score(self.adapter_name)
                is_dict[n] = ipt_score
                all_is.append(ipt_score)

        # Calculate the increasing threshold
        increase_threshold = self.compute_threshold(global_step, all_is)
        with torch.no_grad():
            curr_sum_rank = 0
            sum_param = 0
            new_param_list = []
            add_r = self.incre_rank_num
            for n, layer in model.named_modules():
                if isinstance(layer, TriAdaptLoraLayer) and self.adapter_name in layer.lora_A.keys():
                    # print("===================param name :{}====================".format(n))
                    # size = len(layer.lora_C_row[self.adapter_name])
                    # print("===================self.adapter_name :{}===================".format(self.adapter_name))
                    # print("=========param name :{}==========lora_C_row[{}] :{}===================".format(n, self.adapter_name, layer.lora_C_row[self.adapter_name]))
                    # for i in range(size):
                    #     print("i : ", i)
                    #     print("=========param name :{}==========lora_C_row[{}][{}] :{}===================".format(n, self.adapter_name, i, layer.lora_C_row[self.adapter_name][i]))
                    #     print("=========param name :{}==========lora_C_col[{}][{}] :{}===================".format(n, self.adapter_name, i, layer.lora_C_col[self.adapter_name][i]))
                    #     print("=========param name :{}==========layer.lora_A[{}][{}] :{}===================".format(n, self.adapter_name, i, layer.lora_A[self.adapter_name][i]))
                    #     print("=========param name :{}==========layer.lora_B[{}][{}] :{}===================".format(n, self.adapter_name, i, layer.lora_B[self.adapter_name][i]))
                    if is_dict[n] >= increase_threshold:
                        self.total_rank += add_r
                        # Parameter matrix expansion.
                        layer.add_parameters(add_r, self.adapter_name)
                        # Take the last add_r column of the lists `lora_A`, `lora_B`, `lora_C_row`, and `lora_C_col`.
                        new_param_list.append(layer.lora_A[self.adapter_name][-1])
                        new_param_list.append(layer.lora_B[self.adapter_name][-1])
                        new_param_list.append(layer.lora_C_row[self.adapter_name][-1])
                        new_param_list.append(layer.lora_C_col[self.adapter_name][-1])
                        print("The lora parameters rank of {} increased by {}".format(n, add_r))

                    ranknum = layer.ranknum[self.adapter_name]
                    print("Ranknum/%s" % (n,), ranknum, self.global_step)
                    self.rank_pattern[n] = ranknum
                    curr_sum_rank += ranknum
                    sum_param += ranknum * self.shape_dict[n + ".lora_A"][1]
                    sum_param += ranknum * self.shape_dict[n + ".lora_B"][0]
                    sum_param += 2 * ranknum * ranknum
            print("888888888888888888888888888  Enter optimizer.add_param_group  888888888888888888888888888888888")
            # print("=====================new_param_list :{}=====================".format(new_param_list))
            optimizer.add_param_group({'params': new_param_list, "weight_decay": self.weight_decay, })
            print("Budget/total_rank", curr_sum_rank, self.global_step)
            print("Budget/increase_threshold", increase_threshold, self.global_step)
            print("Budget/sum_param", sum_param, self.global_step)

        return increase_threshold

    def update_and_increase(self, model, global_step, optimizer):
        self.global_step = global_step
        increase_threshold = None
        if self.total_rank < self.target_total_rank:
            if global_step > self.init_warmup and global_step % self.incre_interval == 0:
                increase_threshold = self.increase_to_target_rank(model, optimizer, global_step)
        return self.top_k, increase_threshold

