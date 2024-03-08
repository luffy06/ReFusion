import re, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
logger = logging.getLogger(__name__)

from typing import List, Optional, Tuple, Union

COMMON_LAYERS_PATTERN = ["layers", "h", "block", "blocks"]

class RetrievalOutput(nn.Module):
    def __init__(self, in_features, out_features, layer_norm_eps=1e-5, dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(in_features, out_features)
        self.LayerNorm = nn.LayerNorm(out_features, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(inputs)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + inputs)
        return hidden_states


class RetrievalLayer(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        retriever, 
        fusion_strategy, 
        retrieve_texts,
        layer_name=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.retriever = retriever
        self.fusion_strategy = fusion_strategy
        self.retrieve_texts = retrieve_texts
        self.layer_name = layer_name
        
        # # Initialize weights to transfer the retrieval embeddings
        # self.pert_output = RetrievalOutput(self.retriever.retrieval_dim, self.out_features)

        # Initialize parameters to mask some retrievals
        self.mask = torch.Tensor(self.retriever.topk, 1)
        nn.init.xavier_uniform_(self.mask)
        self.mask = nn.Parameter(self.mask, requires_grad=True)

        # Initialize parameters for the ordered mask
        self.beta = torch.Tensor(self.retriever.topk, self.retriever.retrieval_dim)
        nn.init.xavier_uniform_(self.beta)
        self.beta = nn.Parameter(self.beta, requires_grad=True)
        self.lb = -5.0
        self.ub = 5.0
        self.tau = 5.0

    def update_layer(self, retriever, fusion_strategy, retrieve_texts, layer_name=None):
        self.retriever = retriever
        self.fusion_strategy = fusion_strategy
        self.retrieve_texts = retrieve_texts
        self.layer_name = layer_name
        self.to(self.weight.device)
    
    def _get_ordered_mask(self):
        clamped_beta = F.sigmoid(torch.clamp(self.beta, self.lb, self.ub))
        qz = torch.cumprod(clamped_beta, dim=0) * (1 - clamped_beta)
        sample = F.gumbel_softmax(qz, tau=self.tau, hard=False)
        # if self.training:
        #     sample = F.gumbel_softmax(qz, tau=self.tau, hard=False)
        # else:
        #     sample = (qz / self.tau).softmax(-1)
        ordered_mask = torch.flip(sample.cumsum(dim=0), dims=[0])
        return ordered_mask

    def _get_neighbors(self, x, pos=0):
        if self.retrieve_texts:
            neighbors = self.retriever.fetch_from_cache()
        else:
            queries = x[:, pos, :].squeeze(1).detach().cpu().numpy()
            neighbors = self.retriever.search(queries)
        if isinstance(neighbors, np.ndarray):
            neighbors = torch.tensor(neighbors).to(x.device)
        if neighbors.shape[1] == self.retriever.topk * 2:
            neighbors = (neighbors[:, :self.retriever.topk, :] + neighbors[:, self.retriever.topk:, :]) / 2
        return neighbors

    def _add_cls(self, x, y, cls_pos=0, mask=None):
        neighbors = self._get_neighbors(x, cls_pos)
        neighbors = mask * neighbors if mask != None else neighbors
        neighbors = torch.mean(neighbors, dim=1, keepdim=False)
        neighbors = neighbors.unsqueeze(1).repeat(1, y.shape[1], 1)
        neighbors[:, 0:cls_pos, :] = 0
        neighbors[:, cls_pos+1:, :] = 0
        result = y + neighbors
        return result

    def _mask_add_cls(self, x, y, cls_pos=0):
        mask = F.softmax(self.mask, dim=0)
        return self._add_cls(x, y, cls_pos=cls_pos, mask=mask)

    def _ordered_mask_add_cls(self, x, y, cls_pos=0):
        mask = self._get_ordered_mask()
        return self._add_cls(x, y, cls_pos=cls_pos, mask=mask)

    def _residual_add_cls(self, x, y, cls_pos=0, mask=None):
        neighbors = self._get_neighbors(x, cls_pos)
        neighbors = mask * neighbors if mask != None else neighbors
        neighbors = self.pert_output(neighbors)
        neighbors = torch.mean(neighbors, dim=1, keepdim=False)
        neighbors = neighbors.unsqueeze(1).repeat(1, y.shape[1], 1)
        neighbors[:, 0:cls_pos, :] = 0
        neighbors[:, cls_pos+1:, :] = 0
        result = y + neighbors
        return result

    def _mask_residual_add_cls(self, x, y, cls_pos=0):
        mask = F.softmax(self.mask, dim=0)
        return self._residual_add_cls(x, y, cls_pos, mask)

    def _ordered_mask_residual_add_cls(self, x, y, cls_pos=0):
        mask = self._get_ordered_mask()
        return self._residual_add_cls(x, y, cls_pos=cls_pos, mask=mask)

    def fuse_retrieval(self, x, y, cls_pos=0):
        if self.fusion_strategy == 'add_cls':
            result = self._add_cls(x, y, cls_pos=cls_pos)
        elif self.fusion_strategy == 'mask_add_cls':
            result = self._mask_add_cls(x, y, cls_pos=cls_pos)
        elif self.fusion_strategy == 'ordered_mask_add_cls':
            result = self._ordered_mask_add_cls(x, y, cls_pos=cls_pos)
        elif self.fusion_strategy == 'residual_add_cls':
            result = self._residual_add_cls(x, y, cls_pos=cls_pos)
        elif self.fusion_strategy == 'mask_residual_add_cls':
            result = self._mask_residual_add_cls(x, y, cls_pos=cls_pos)
        elif self.fusion_strategy == 'ordered_mask_residual_add_cls':
            result = self._ordered_mask_residual_add_cls(x, y, cls_pos=cls_pos)
        else:
            raise NotImplemented
        return result


class RetrievalLinear(RetrievalLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight,
        bias,
        retriever, 
        fusion_strategy, 
        retrieve_texts,
        layer_name=None,
    ):
        super().__init__(
            in_features=in_features, 
            out_features=out_features, 
            retriever=retriever,
            fusion_strategy=fusion_strategy,
            retrieve_texts=retrieve_texts,
            layer_name=layer_name,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.weight = weight
        self.bias = bias

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if '_x' in self.fusion_strategy:
            y = self.fuse_retrieval(x, x)
            result = F.linear(y, self.weight, bias=self.bias)
        else:
            y = F.linear(x, self.weight, bias=self.bias)
            result = self.fuse_retrieval(x, y)
        result = result.to(previous_dtype)
        return result


class MixedLinear(RetrievalLayer):
    # PERT implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight,
        bias,
        retriever, 
        fusion_strategy, 
        retrieve_texts,
        submodules: Optional[List[str]] = None,
        layer_name=None,
    ):
        super().__init__(
            in_features=in_features, 
            out_features=out_features, 
            retriever=retriever,
            fusion_strategy=fusion_strategy,
            retrieve_texts=retrieve_texts,
            layer_name=layer_name,
        )

        self.submodules = submodules if submodules != None else ['identity']
        self.linears = nn.ModuleList([nn.Linear(in_features, out_features) for i in range(len(self.submodules))])
        for linear in self.linears:
            linear.weight = weight
            linear.bias = bias

        self.arch_para = torch.Tensor(len(self.submodules), 1)
        nn.init.xavier_uniform_(self.arch_para)
        self.arch_para = nn.Parameter(self.arch_para)
        self.arch_para.requires_grad = False

    def __del__(self):
        norm_arch_para = F.softmax(self.arch_para, dim=0)
        module_index = torch.argmax(norm_arch_para, dim=0)
        module_name = self.submodules[module_index]
        logger.info(f"Layer {self.layer_name}")
        logger.info(f"Arch. Para. {norm_arch_para.detach().cpu().numpy().squeeze()}")
        logger.info(f"Choose {module_name}")

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        def module_forward(x, module_index, module_name, arch_weight):
            y = F.linear(x, self.linears[module_index].weight, bias=self.linears[module_index].bias)
            if module_name == 'identity':
                result = y * arch_weight
            elif module_name == 'add_cls':
                z = self._add_cls(x, y)
                result = z * arch_weight
            elif module_name == 'mask_add_cls':
                z = self._mask_add_cls(x, y)
                result = z * arch_weight
            elif module_name == 'ordered_mask_add_cls':
                z = self._ordered_mask_add_cls(x, y)
                result = z * arch_weight
            elif module_name == 'residual_add_cls':
                z = self._residual_add_cls(x, y)
                result = z * arch_weight
            elif module_name == 'mask_residual_add_cls':
                z = self._mask_residual_add_cls(x, y)
                result = z * arch_weight
            elif module_name == 'ordered_mask_residual_add_cls':
                z = self._ordered_mask_residual_add_cls(x, y)
                result = z * arch_weight
            else:
                result = torch.zeros_like(y).to(x.device)
            return result

        norm_arch_para = F.softmax(self.arch_para, dim=0)
        if self.training:
            result = torch.zeros(x.shape[0], x.shape[1], self.out_features).to(x.device)
            for module_index, module_name in enumerate(self.submodules):
                result += module_forward(x, module_index, module_name, norm_arch_para[module_index])
        else:
            module_index = torch.argmax(norm_arch_para, dim=0)
            module_name = self.submodules[module_index]
            result = module_forward(x, module_index, module_name, 1)
        result = result.to(previous_dtype)
        return result


class RetrievalWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.enable_retrieval = False
        self.retrieve_texts = False
        self.enable_nas = False

    def add_retrieval(self, config, retriever, query_encoder):
        self.enable_retrieval = True
        self.enable_nas = config.retrieval_mode == 'nas'
        self.nas_modules = config.nas_modules if self.enable_nas else []
        self.retriever = retriever
        self.fusion_strategy = config.fusion_strategy
        self.retrieve_texts = config.retrieve_texts

        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.named_modules()]
        for key in key_list:
            if not self._check_target_module_exists(config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = self._get_submodules(key)

            if isinstance(target, RetrievalLayer):
                target.update_layer(
                    self.retriever,
                    self.fusion_strategy,
                    self.retrieve_texts,
                    layer_name=key
                )
            else:
                new_module = self._create_new_module(target, key)
                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )
        if not config.full_training:
            self.mark_only_pert_as_trainable()

        # Should be after the replacing module process, otherwise the modules inside query encoder might be replaced
        if self.retrieve_texts:
            self.query_encoder = query_encoder

    def _check_target_module_exists(self, config, key):
        if isinstance(config.target_modules, str):
            target_module_found = re.fullmatch(config.target_modules, key)
        else:
            target_module_found = any(key.split('.')[-1] == target_key for target_key in config.target_modules)
            is_using_layer_indexes = getattr(config, "layers_to_transform", None) is not None
            layer_indexing_pattern = getattr(config, "layers_pattern", None)

            if is_using_layer_indexes and target_module_found:
                layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

                for pattern in layers_pattern:
                    layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
                    if layer_index is not None:
                        layer_index = int(layer_index.group(1))
                        if isinstance(config.layers_to_transform, int):
                            target_module_found = layer_index == config.layers_to_transform
                        else:
                            target_module_found = layer_index in config.layers_to_transform
                        break
                    else:
                        target_module_found = False
        return target_module_found
    
    def _get_submodules(self, key):
        parent = self.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.get_submodule(key)
        return parent, target, target_name

    def _create_new_module(self, target, layer_name=None):
        if isinstance(target, torch.nn.Linear):
            if self.enable_nas:
                new_module = MixedLinear(
                    in_features=target.in_features, 
                    out_features=target.out_features, 
                    weight=target.weight,
                    bias=target.bias, 
                    retriever=self.retriever,
                    fusion_strategy=self.fusion_strategy,
                    retrieve_texts=self.retrieve_texts,
                    submodules=self.nas_modules,
                    layer_name=layer_name,
                )
            else:
                new_module = RetrievalLinear(
                    in_features=target.in_features, 
                    out_features=target.out_features, 
                    weight=target.weight,
                    bias=target.bias, 
                    retriever=self.retriever,
                    fusion_strategy=self.fusion_strategy,
                    retrieve_texts=self.retrieve_texts,
                    layer_name=layer_name,
                )
        else:
            raise ValueError(
                f"Target module {target} is not supported. "
                f"Currently, only `torch.nn.Linear` are supported."
            )

        return new_module

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        old_device = None
        for n, p in old_module.named_parameters():
            old_device = p.device
        if old_device != None:
            for name, module in new_module.named_modules():
                module.to(old_device)

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        logger.info(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}")

    def mark_only_pert_as_trainable(self):
        for n, p in self.named_parameters():
            if "pert_" not in n:
                p.requires_grad = False
