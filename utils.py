import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, query_mask=None, key_mask=None, mask=None, return_attn=False):
    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)

    NEG = -1e9


    if key_mask is not None:
        scores = scores.masked_fill(~key_mask.unsqueeze(1), NEG)


    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        scores = scores.masked_fill(~mask, NEG)

    attn = F.softmax(scores, dim=-1)


    if query_mask is not None:
        qm = query_mask.unsqueeze(-1).to(attn.dtype)
        attn = attn * qm

    out = torch.matmul(attn, v)

    if query_mask is not None:
        out = out * qm

    if return_attn:
        return out, attn
    return out
