"""Measure dense dynamic KV payload using matched Llama models."""
import argparse
import json
import platform
from time import perf_counter

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM


def payload_bytes(cache):
    return sum(
        tensor.numel() * tensor.element_size()
        for layer in cache.layers
        for tensor in (layer.keys, layer.values)
    )


def tiny_model(kv_heads, max_length):
    torch.manual_seed(226)
    config = LlamaConfig(
        vocab_size=256, hidden_size=256, intermediate_size=512,
        num_hidden_layers=4, num_attention_heads=8,
        num_key_value_heads=kv_heads, max_position_embeddings=max_length,
    )
    return LlamaForCausalLM(config).eval()


@torch.inference_mode()
def measure(model, ids, synchronize):
    synchronize()
    start = perf_counter()
    cache = model.model(input_ids=ids, use_cache=True).past_key_values
    synchronize()
    return cache, perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int, default=[128, 512, 1024])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    args = parser.parse_args()
    if min(*args.lengths, args.batch, args.repeats, args.threads) < 1:
        parser.error("lengths, batch, repeats and threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    synchronize = torch.cuda.synchronize if args.device == "cuda" else lambda: None
    inputs = torch.randint(
        0, 256, (args.batch, max(args.lengths)),
        generator=torch.Generator().manual_seed(226),
    ).to(args.device)

    for kv_heads in (8, 2):
        model = tiny_model(kv_heads, max(args.lengths)).to(device=args.device, dtype=dtype)
        config = model.config
        for length in args.lengths:
            ids = inputs[:, :length].contiguous()
            measure(model, ids, synchronize)
            for repeat in range(args.repeats):
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    before = torch.cuda.memory_allocated()
                cache, elapsed = measure(model, ids, synchronize)
                peak = torch.cuda.max_memory_allocated() if args.device == "cuda" else None
                size = payload_bytes(cache)
                expected = (
                    2 * args.batch * config.num_hidden_layers * length
                    * kv_heads * config.head_dim * torch.empty((), dtype=dtype).element_size()
                )
                assert cache.get_seq_length() == length and size == expected
                print(json.dumps(dict(
                    architecture="MHA" if kv_heads == 8 else "GQA",
                    model="random_init_llama", seed=226, batch=args.batch, repeat=repeat,
                    seq_len=length, layers=config.num_hidden_layers,
                    q_heads=config.num_attention_heads, kv_heads=kv_heads, head_dim=config.head_dim,
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    dtype=args.dtype, device=args.device, threads=args.threads,
                    hardware=torch.cuda.get_device_name() if peak is not None else platform.machine(),
                    os=platform.platform(), torch=torch.__version__, transformers=transformers.__version__,
                    attention_backend=config._attn_implementation, kv_bytes=size,
                    predicted_kv_bytes=expected, kv_gb=size / 1e9,
                    prefill_ms=elapsed * 1000, cuda_peak_allocated_bytes=peak,
                    cuda_peak_above_pre_forward_bytes=peak - before if peak is not None else None,
                )), flush=True)
                del cache
        del model


if __name__ == "__main__":
    main()
