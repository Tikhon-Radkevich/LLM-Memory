import math

import torch
from torch import nn
from transformers import AutoModel, MPNetConfig, MPNetModel
from transformers.models.mpnet.modeling_mpnet import MPNetAttention, MPNetSelfAttention


class MemoryAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int = 64):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank)
        self.up = nn.Linear(rank, hidden_size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class MemoryMPNetConfig(MPNetConfig):
    def __init__(self, memory_size: int = 16, adapter_rank: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.memory_size = memory_size
        self.adapter_rank = adapter_rank


class MemoryMPNetSelfAttention(MPNetSelfAttention):
    def __init__(self, config: MemoryMPNetConfig):
        super().__init__(config)
        self.memory_adapter = MemoryAdapter(
            config.hidden_size,
            rank=config.adapter_rank,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,  # ignored: was computed for seq_len×seq_len, memory tokens are position-agnostic
        output_attentions=False,
        **kwargs,
    ):
        memory = kwargs.get("memory")  # [mem_size, H] or [B, mem_size, H]

        if memory is not None:
            if memory.dim() == 2:
                memory = memory.unsqueeze(0).expand(hidden_states.size(0), -1, -1)
            mem_size = memory.size(1)
            combined = torch.cat([memory, hidden_states], dim=1)  # [B, mem+seq, H]
        else:
            mem_size = 0
            combined = hidden_states

        input_shape = combined.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)

        q = self.q(combined).view(hidden_shape).transpose(1, 2)
        k = self.k(combined).view(hidden_shape).transpose(1, 2)
        v = self.v(combined).view(hidden_shape).transpose(1, 2)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None and mem_size > 0:
            # Memory tokens are on the left in combined, so prepend zeros for their key columns
            mem_key_mask = torch.zeros(
                attention_mask.size(0), 1, 1, mem_size,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_scores = attention_scores + torch.cat([mem_key_mask, attention_mask], dim=-1)
        elif attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        c = torch.matmul(attention_probs, v)
        c = c.permute(0, 2, 1, 3).contiguous()
        c = c.view(*c.size()[:-2], self.all_head_size)  # [B, seq+mem, H]

        o = self.o(c)

        input_output = o[:, mem_size:, :]  # [B, seq_len, H] — returned as normal output

        if mem_size > 0:
            memory_output = o[:, :mem_size, :]          # [B, mem_size, H]
            self._updated_memory = self.memory_adapter(memory_output)
        else:
            self._updated_memory = None

        outputs = (input_output, attention_probs) if output_attentions else (input_output,)
        return outputs


class MemoryMPNetAttention(MPNetAttention):
    def __init__(self, config: MemoryMPNetConfig):
        super().__init__(config)
        self.attn = MemoryMPNetSelfAttention(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        output_attentions=False,
        **kwargs,
    ):
        self_outputs = self.attn(
            hidden_states,
            attention_mask,
            position_bias,
            output_attentions=output_attentions,
            **kwargs,  # passes memory through
        )
        # self_outputs[0] is input_output [B, seq_len, H]; residual with hidden_states is valid
        attention_output = self.LayerNorm(self.dropout(self_outputs[0]) + hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class MemoryMPNetModel(nn.Module):
    def __init__(self, base_model: MPNetModel, config=None):
        super().__init__()
        self.mpnet = base_model
        cfg = config or base_model.config

        last_layer = self.mpnet.encoder.layer[-1]
        new_attn = MemoryMPNetAttention(cfg)
        new_attn.load_state_dict(last_layer.attention.state_dict(), strict=False)
        last_layer.attention = new_attn

    def forward(self, input_ids, memory, attention_mask=None):
        out = self.mpnet(input_ids=input_ids, attention_mask=attention_mask, memory=memory)
        sequence_output = out[0]  # [B, seq_len, H]
        updated_memory = self.mpnet.encoder.layer[-1].attention.attn._updated_memory  # [B, mem_size, H]
        return sequence_output, updated_memory.squeeze(0)  # squeeze batch dim (B=1)


def build_memory_mpnet(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    adapter_rank: int = 64,
    memory_size: int = 16,
) -> MemoryMPNetModel:
    base = AutoModel.from_pretrained(model_name)
    config = MemoryMPNetConfig.from_pretrained(
        model_name,
        memory_size=memory_size,
        adapter_rank=adapter_rank,
    )
    return MemoryMPNetModel(base, config=config)
