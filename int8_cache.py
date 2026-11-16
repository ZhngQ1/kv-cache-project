"""CPU research baseline with INT8 storage and FP32 attention."""
import torch
from transformers.cache_utils import Cache, DynamicLayer


class Int8Layer(DynamicLayer):
    @staticmethod
    def quantize(states):
        scale = (states.abs().amax(dim=-1, keepdim=True) / 127).clamp_min(
            torch.finfo(states.dtype).tiny
        )
        return (states / scale).round().clamp(-127, 127).to(torch.int8), scale

    def update(self, key_states, value_states, cache_kwargs=None):
        keys, key_scales = self.quantize(key_states)
        values, value_scales = self.quantize(value_states)
        if self.is_initialized:
            keys = torch.cat((self.keys, keys), dim=-2)
            values = torch.cat((self.values, values), dim=-2)
            key_scales = torch.cat((self.key_scales, key_scales), dim=-2)
            value_scales = torch.cat((self.value_scales, value_scales), dim=-2)
        else:
            self.lazy_initialization(key_states)
        self.keys, self.values = keys, values
        self.key_scales, self.value_scales = key_scales, value_scales
        return keys * key_scales, values * value_scales


def int8_cache(layer_count):
    return Cache(layers=[Int8Layer() for _ in range(layer_count)])


def scale_bytes(cache):
    return sum(
        tensor.numel() * tensor.element_size()
        for layer in cache.layers
        for tensor in (layer.key_scales, layer.value_scales)
    )
