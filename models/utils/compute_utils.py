import torch
import bitsandbytes.functional as BF

def hidden_to_head_shape(x: torch.Tensor, num_heads: int):
    bsz, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads
    return x.reshape(bsz, seq_len, num_heads, head_dim).transpose(1, 2)


def head_to_hidden_shape(x: torch.Tensor):
    bsz, _, seq_len, _ = x.shape
    return x.transpose(1, 2).reshape(bsz, seq_len, -1)


def lora_forward(w, w_quant_state, w_lora_a, w_lora_b, b, x):
    w_dequant = BF.dequantize_nf4(w, w_quant_state).t()
    x = x.to(w_dequant.dtype)
    x_main = x @ w_dequant + b.to(w_dequant.dtype) if b is not None else x @ w_dequant
    x_lora_a = x @ w_lora_a.to(w_dequant.dtype)
    x_lora = x_lora_a @ w_lora_b.to(w_dequant.dtype)
    x = x_main + x_lora
    return x, x_main, x_lora_a


def lora_backward(w, w_quant_state, w_lora_a, w_lora_b, x, x_lora_a, grad_y):
    w_dequant = BF.dequantize_nf4(w, w_quant_state).t()
    grad_medium = grad_y.to(w_dequant.dtype) @ w_lora_b.mT
    w_lora_a, w_lora_b = w_lora_a.to(w_dequant.dtype), w_lora_b.to(w_dequant.dtype)
    grad_w_lora_a = x.to(w_dequant.dtype).mT @ (grad_medium)
    grad_w_lora_b = (x_lora_a.mT @ grad_y.to(w_lora_b.dtype))
    grad_x = grad_y.to(w_dequant.dtype) @ w_dequant.T 
    grad_x += (grad_medium @ w_lora_a.T)
    return grad_w_lora_a, grad_w_lora_b, grad_x


def repeat_kv(hidden_states: torch.Tensor, n_rep: int):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def repeat_kv_backward(grad_output: torch.Tensor, n_rep: int):
    batch, expand_num_key_value_heads, slen, head_dim = grad_output.shape
    num_key_value_heads = expand_num_key_value_heads // n_rep
    grad_output = grad_output.reshape(batch, num_key_value_heads, n_rep, slen, head_dim)
    return grad_output.sum(dim=2)