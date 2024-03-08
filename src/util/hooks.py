import torch
import torch.nn as nn

from thop import profile

from models.layers import RetrievalLinear, MixedLinear
from models.roberta import RetrievalAugmentedRoberta
from transformers.activations import GELUActivation
from transformers.models.roberta.modeling_roberta import (
    RobertaSelfAttention,
    RobertaEmbeddings,
    RobertaSelfOutput,
    RobertaAttention,
    RobertaIntermediate,
    RobertaOutput,
    RobertaLayer,
)

import logging
logger = logging.getLogger(__name__)

    
def zero_ops(m, x, y):
    m.total_ops += torch.DoubleTensor([int(0)])

def count_normalization(m, x, y):
    x = x[0]
    # bn is by default fused in inference
    flops = torch.DoubleTensor([2 * x.numel()])
    if (getattr(m, 'affine', False) or getattr(m, 'elementwise_affine', False)):
        flops *= 2
    m.total_ops += flops

def count_linear(m, x, y):
    x = x[0]
    assert x.shape[0] == 1
    total_mul = 2 * x.shape[1] * m.in_features * m.out_features
    m.total_ops += torch.DoubleTensor([int(total_mul)])

def count_retrieval_linear(m, x, y):
    x = x[0]
    assert x.shape[0] == 1
    total_mul = 2 * x.shape[1] * m.in_features * m.out_features
    m.total_ops += torch.DoubleTensor([int(total_mul)])
    retrieval_add = (m.retriever.topk - 1) * m.retriever.retrieval_dim
    if m.fusion_strategy == 'mask_add_cls' or m.fusion_strategy == 'ordered_mask_add_cls':
        retrieval_add += m.retriever.topk * m.retriever.retrieval_dim
    m.total_ops += torch.DoubleTensor([int(retrieval_add)])
    m.total_ops += torch.DoubleTensor([int(m.in_features)])

def count_self_attention(m, x, y):
    x = x[0]
    if isinstance(m.query, nn.Linear):
        total_mul = 2 * x.shape[1] * m.query.in_features * m.query.out_features
        m.total_ops += torch.DoubleTensor([int(total_mul)])
    elif isinstance(m.query, RetrievalLinear):
        total_mul = 2 * x.shape[1] * m.query.in_features * m.query.out_features
        m.total_ops += torch.DoubleTensor([int(total_mul)])
        retrieval_add = (m.query.retriever.topk - 1) * m.query.retriever.retrieval_dim
        if m.query.fusion_strategy == 'mask_add_cls' or m.query.fusion_strategy == 'ordered_mask_add_cls':
            retrieval_add += m.query.retriever.topk * m.query.retriever.retrieval_dim
        m.total_ops += torch.DoubleTensor([int(retrieval_add)])
        m.total_ops += torch.DoubleTensor([int(m.query.in_features)])

    if isinstance(m.key, nn.Linear):
        total_mul = 2 * x.shape[1] * m.key.in_features * m.key.out_features
        m.total_ops += torch.DoubleTensor([int(total_mul)])
    elif isinstance(m.key, RetrievalLinear):
        total_mul = 2 * x.shape[1] * m.key.in_features * m.key.out_features
        m.total_ops += torch.DoubleTensor([int(total_mul)])
        retrieval_add = (m.key.retriever.topk - 1) * m.key.retriever.retrieval_dim
        if m.key.fusion_strategy == 'mask_add_cls' or m.key.fusion_strategy == 'ordered_mask_add_cls':
            retrieval_add += m.key.retriever.topk * m.key.retriever.retrieval_dim
        m.total_ops += torch.DoubleTensor([int(retrieval_add)])
        m.total_ops += torch.DoubleTensor([int(m.key.in_features)])

    if isinstance(m.value, nn.Linear):
        total_mul = 2 * x.shape[1] * m.value.in_features * m.value.out_features
        m.total_ops += torch.DoubleTensor([int(total_mul)])
    elif isinstance(m.value, RetrievalLinear):
        total_mul = 2 * x.shape[1] * m.value.in_features * m.value.out_features
        m.total_ops += torch.DoubleTensor([int(total_mul)])
        retrieval_add = (m.value.retriever.topk - 1) * m.value.retriever.retrieval_dim
        if m.value.fusion_strategy == 'mask_add_cls' or m.value.fusion_strategy == 'ordered_mask_add_cls':
            retrieval_add += m.value.retriever.topk * m.value.retriever.retrieval_dim
        m.total_ops += torch.DoubleTensor([int(retrieval_add)])
        m.total_ops += torch.DoubleTensor([int(m.value.in_features)])

    total_mul = 2 * x.shape[1] * m.query.out_features * x.shape[1]
    total_mul += 2 * x.shape[1] * x.shape[1] * m.value.out_features
    m.total_ops += torch.DoubleTensor([int(total_mul)])

custom_ops = {
    nn.LayerNorm: count_normalization,
    nn.Linear: count_linear,
    RetrievalLinear: count_retrieval_linear,
    MixedLinear: count_retrieval_linear,
    RobertaSelfAttention: count_self_attention,
}

def count_params(model):
    model.train()
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    logger.info(f'Trainable params {trainable_params / 1e6:.2f} M')
    logger.info(f'All params {all_param / 1e6:.2f} M')

def compute_flops(dataloader, model):
    seq_len = 0
    macs = 0
    params = 0
    for i, batch in enumerate(dataloader):
        assert batch['input_ids'].shape[0] == 1
        seq_len += batch['attention_mask'].sum()
        if i == 0:
            macs, params = profile(
                model, 
                inputs={k: v.to(model.device) if hasattr(v, 'to') else v for k, v in batch.items()}, 
                custom_ops=custom_ops, 
                report_missing=True
            )
    logger.info("***** Statistics *****")
    logger.info(f'FLOPs {macs} {macs / 1e9:.2f} G')
    count_params(model)
    logger.info(f'Seq len {seq_len / len(dataloader.dataset):.2f}')
    mem = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
    logger.info(f'GPU memory {mem:.2f} GB')

if __name__ == "__main__":
    compute_flops()
