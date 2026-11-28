import argparse
import json
import math
import os
import sys
import sysconfig
import time
from pathlib import Path

sys.path.append(".")

from ppl_data import get_test_loader


def build_parser():
    parser = argparse.ArgumentParser(
        description="PPL evaluation on top of vLLM-Ascend."
    )
    parser.add_argument("--model", type=str, required=True, help="HF model path or name")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["wikitext2", "ptb", "ptb-new", "c4", "c4-new"],
        required=True,
        help="Evaluation dataset.",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=2048,
        help="Prompt length for each evaluation window.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of windows per vLLM batch.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional cap on the number of evaluation windows.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for vLLM.")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "fp16", "bf16", "fp32"],
        default="bf16",
        help="Model dtype for vLLM.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--enable-expert-parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable vLLM expert parallelism (required by Dynamic EPLB/MC2).",
    )
    parser.add_argument(
        "--additional-config",
        type=json.loads,
        default=None,
        help="JSON object forwarded to vLLM as additional_config.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override vLLM max_model_len. Defaults to seqlen + 1.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help=(
            "vLLM memory utilization ratio. Defaults to 0.90 to leave "
            "headroom for FP32 prompt-logprob computation."
        ),
    )
    parser.add_argument(
        "--swap-space",
        type=float,
        default=16.0,
        help="vLLM swap space in GiB.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to vLLM.",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable eager execution. Defaults to true for lower startup memory.",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable vLLM prefix caching. Defaults to false to save memory.",
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable chunked prefill. Defaults to false to reduce startup pressure.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Override vLLM max_num_batched_tokens. Defaults to max_model_len.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="Maximum number of concurrent sequences tracked by vLLM.",
    )
    parser.add_argument(
        "--disable-sliding-window",
        action="store_true",
        help="Pass disable_sliding_window=True to vLLM when supported.",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default=None,
        help="Optional kv_cache_dtype passed through to vLLM.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["group_quant"],
        help="Optional vLLM quantization backend.",
    )
    parser.add_argument(
        "--weight-bits",
        type=int,
        default=8,
        help="Weight quantization bits for custom group_quant backend.",
    )
    parser.add_argument(
        "--act-bits",
        type=int,
        default=8,
        help="Activation quantization bits for custom group_quant backend.",
    )
    parser.add_argument(
        "--residual-bits",
        type=int,
        default=4,
        help="Residual exponent bits for custom group_quant backend.",
    )
    parser.add_argument(
        "--residual-group",
        type=int,
        default=32,
        help="Group size for custom group_quant backend.",
    )
    parser.add_argument(
        "--disable-weight-quant",
        action="store_true",
        help="Disable weight fake quantization in the custom group_quant backend.",
    )
    parser.add_argument(
        "--disable-activation-quant",
        action="store_true",
        help="Disable activation fake quantization in the custom group_quant backend.",
    )
    parser.add_argument(
        "--quantize-moe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable custom GroupQuant MoE path for fused experts.",
    )
    parser.add_argument(
        "--quantize-linear",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Quantize non-MoE Linear layers. Disable this to keep attention, "
            "router, and LM-head linears on their optimized BF16 path while "
            "retaining GroupQuant MoE kernels."
        ),
    )
    parser.add_argument(
        "--expert-gmm2-op",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable QuantMatmulExpertKGrouped for GroupQuant MoE "
            "GMM2. If omitted, preserve GROUP_QUANT_USE_EXPERT_GMM2_OP."
        ),
    )
    parser.add_argument(
        "--gmm1-vecin-fullk",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable the fused GroupQuant GMM1→INT8+scale path. "
            "It directly feeds QuantMatmulExpertKGrouped and is valid only "
            "for its supported full-K MoE shapes. If omitted, preserve "
            "GROUP_QUANT_USE_GMM1_VECIN_FULLK."
        ),
    )
    return parser


def resolve_vllm_dtype(dtype_name: str) -> str:
    if dtype_name == "fp16":
        return "float16"
    if dtype_name == "bf16":
        return "bfloat16"
    if dtype_name == "fp32":
        return "float32"
    return "auto"


def validate_args(args):
    expert_gmm2_enabled = (
        args.expert_gmm2_op
        if args.expert_gmm2_op is not None
        else os.getenv("GROUP_QUANT_USE_EXPERT_GMM2_OP", "0") == "1"
    )
    if expert_gmm2_enabled and (
        args.quantization != "group_quant" or not args.quantize_moe
    ):
        raise ValueError(
            "--expert-gmm2-op requires --quantization group_quant and "
            "--quantize-moe."
        )
    if args.gmm1_vecin_fullk is True and not expert_gmm2_enabled:
        raise ValueError(
            "--gmm1-vecin-fullk requires --expert-gmm2-op so its INT8 "
            "activation and per-token scale are consumed without a second "
            "dynamic-quant operation."
        )


def configure_expert_gmm2_op(args):
    if args.expert_gmm2_op is False:
        os.environ.pop("GROUP_QUANT_USE_EXPERT_GMM2_OP", None)
        return

    enabled = (
        args.expert_gmm2_op is True
        or os.getenv("GROUP_QUANT_USE_EXPERT_GMM2_OP", "0") == "1"
    )
    if not enabled:
        return

    repo_root = Path(__file__).resolve().parent
    vendor_root = (
        repo_root
        / "custom_op/build_root/install/vendors/custom_transformer"
    )
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    adapter_candidates = [
        candidate
        for candidate in sorted(
        (repo_root / "custom_op/quant_matmul_expert_k_grouped/build").glob(
            "lib.*/quant_matmul_expert_k_grouped_torch*.so"
        )
        )
        if extension_suffix and candidate.name.endswith(extension_suffix)
    ]
    op_api = vendor_root / "op_api/lib/libcust_opapi.so"
    if not op_api.is_file() or not adapter_candidates:
        raise RuntimeError(
            "QuantMatmulExpertKGrouped is not built. Run: "
            "bash custom_op/scripts/build_quant_matmul_expert_k_grouped.sh"
        )

    vendor = str(vendor_root)
    custom_opp = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    if vendor not in custom_opp.split(":"):
        raise RuntimeError(
            "Custom OPP environment is not loaded. Run: source "
            "env_vllm_ascend.sh"
        )
    os.environ["GROUP_QUANT_USE_EXPERT_GMM2_OP"] = "1"
    os.environ["GROUP_QUANT_CUSTOM_OPAPI_PATH"] = str(op_api)
    print(
        "GroupQuant GMM2 custom op: enabled, adapter="
        f"{adapter_candidates[-1]}"
    )


def configure_gmm1_vecin_fullk(args):
    if args.gmm1_vecin_fullk is False:
        os.environ.pop("GROUP_QUANT_USE_GMM1_VECIN_FULLK", None)
        return

    enabled = (
        args.gmm1_vecin_fullk is True
        or os.getenv("GROUP_QUANT_USE_GMM1_VECIN_FULLK", "0") == "1"
    )
    if not enabled:
        return

    repo_root = Path(__file__).resolve().parent
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    adapters = [
        candidate
        for candidate in sorted(
            (repo_root / "custom_op/gmm1_vecin_fullk/build").glob(
                "lib.*/gmm1_vecin_fullk_torch*.so"
            )
        )
        if extension_suffix and candidate.name.endswith(extension_suffix)
    ]
    op_api = (
        repo_root
        / "custom_op/build_root/install/vendors/custom_transformer/op_api/lib/libcust_opapi.so"
    )
    if not op_api.is_file() or not adapters:
        raise RuntimeError(
            "Gmm1VecinFullK is not built. Run: cmake --build custom_op/build "
            "--target registry_ops --parallel 8"
        )
    os.environ["GROUP_QUANT_USE_GMM1_VECIN_FULLK"] = "1"
    os.environ["GROUP_QUANT_CUSTOM_OPAPI_PATH"] = str(op_api)
    print(
        "GroupQuant fused GMM1→GMM2 activation path: enabled, adapter="
        f"{adapters[-1]}"
    )


def chunk_list(values, chunk_size: int):
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def build_eval_windows(testenc, seqlen: int, max_windows: int | None):
    flat_tokens = testenc.reshape(-1).tolist()
    total_windows = len(flat_tokens) // seqlen
    if max_windows is not None:
        total_windows = min(total_windows, max_windows)

    windows = []
    for idx in range(total_windows):
        start = idx * seqlen
        end = start + seqlen
        window = flat_tokens[start:end]
        if len(window) == seqlen:
            windows.append(window)
    return windows


def iter_selected_prompt_logprobs(prompt_logprobs, prompt_token_ids):
    if prompt_logprobs is None:
        raise RuntimeError(
            "vLLM did not return prompt_logprobs. Make sure the current "
            "vLLM/vllm-ascend build supports SamplingParams(prompt_logprobs=1)."
        )

    if all(
        hasattr(prompt_logprobs, attr)
        for attr in ("token_ids", "logprobs", "start_indices", "end_indices")
    ):
        token_ids = prompt_logprobs.token_ids
        logprobs = prompt_logprobs.logprobs
        start_indices = prompt_logprobs.start_indices
        end_indices = prompt_logprobs.end_indices
        for pos, token_id in enumerate(prompt_token_ids):
            start = start_indices[pos]
            end = end_indices[pos]
            if start >= end:
                yield None
                continue
            if token_ids[start] != token_id:
                raise RuntimeError(
                    f"Prompt logprob token mismatch at position {pos}: "
                    f"expected {token_id}, got {token_ids[start]}"
                )
            yield float(logprobs[start])
        return

    for pos, token_id in enumerate(prompt_token_ids):
        entry = prompt_logprobs[pos]
        if entry is None:
            yield None
            continue
        if token_id not in entry:
            got = next(iter(entry))
            raise RuntimeError(
                f"Prompt logprob token mismatch at position {pos}: "
                f"expected {token_id}, got {got}"
            )
        value = entry[token_id]
        yield float(getattr(value, "logprob", value))


def compute_batch_nll(outputs, batch_prompt_token_ids):
    batch_nll = 0.0
    batch_predicted_tokens = 0
    for output, prompt_token_ids in zip(outputs, batch_prompt_token_ids):
        prompt_logprobs = list(
            iter_selected_prompt_logprobs(output.prompt_logprobs, prompt_token_ids)
        )
        if len(prompt_logprobs) != len(prompt_token_ids):
            raise RuntimeError(
                "prompt_logprobs length does not match prompt length: "
                f"{len(prompt_logprobs)} vs {len(prompt_token_ids)}"
            )

        for logprob in prompt_logprobs[1:]:
            if logprob is None:
                raise RuntimeError(
                    "Encountered None prompt logprob beyond the first token."
                )
            batch_nll -= logprob
            batch_predicted_tokens += 1

    return batch_nll, batch_predicted_tokens


def load_vllm():
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError(
            "Could not import `vllm`. Activate the vllm-ascend environment "
            "before running this script."
        ) from exc
    return LLM, SamplingParams


def maybe_register_custom_quant_backend(args):
    if args.quantization == "group_quant":
        import Smoothquant.group_quant_config  # noqa: F401
        from vllm.platforms import current_platform

        supported = getattr(current_platform, "supported_quantization", None)
        if supported is not None and "group_quant" not in supported:
            supported.append("group_quant")


def build_llm(args):
    configure_expert_gmm2_op(args)
    configure_gmm1_vecin_fullk(args)
    maybe_register_custom_quant_backend(args)
    LLM, SamplingParams = load_vllm()
    max_model_len = args.max_model_len or (args.seqlen + 1)
    max_num_batched_tokens = args.max_num_batched_tokens or max_model_len

    llm_kwargs = {
        "model": args.model,
        "tokenizer": args.model,
        "dtype": resolve_vllm_dtype(args.dtype),
        "tensor_parallel_size": args.tensor_parallel_size,
        "seed": args.seed,
        "trust_remote_code": args.trust_remote_code,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "swap_space": args.swap_space,
        "enforce_eager": args.enforce_eager,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_chunked_prefill": args.enable_chunked_prefill,
    }
    if args.enable_expert_parallel:
        llm_kwargs["enable_expert_parallel"] = True
    if args.additional_config is not None:
        if not isinstance(args.additional_config, dict):
            raise ValueError("--additional-config must decode to a JSON object.")
        llm_kwargs["additional_config"] = args.additional_config
    if args.quantization is not None:
        llm_kwargs["quantization"] = args.quantization
    if args.quantization == "group_quant":
        llm_kwargs["hf_overrides"] = {
            "quantization_config": {
                "quant_method": "group_quant",
                "weight_bits": args.weight_bits,
                "act_bits": args.act_bits,
                "residual_bits": args.residual_bits,
                "residual_group": args.residual_group,
                "quantize_weights": not args.disable_weight_quant,
                "quantize_activations": not args.disable_activation_quant,
                "quantize_moe": args.quantize_moe,
                "quantize_linear": args.quantize_linear,
            }
        }
    if args.disable_sliding_window:
        llm_kwargs["disable_sliding_window"] = True
    if args.kv_cache_dtype is not None:
        llm_kwargs["kv_cache_dtype"] = args.kv_cache_dtype

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
        detokenize=False,
    )
    return llm, sampling_params


def evaluate_perplexity(args):
    validate_args(args)

    testloader = get_test_loader(args.dataset, seqlen=args.seqlen, model=args.model)
    testenc = testloader.input_ids
    total_tokens = int(testenc.numel())
    windows = build_eval_windows(testenc, args.seqlen, args.max_windows)
    if not windows:
        raise ValueError(
            f"No evaluation windows were built from dataset={args.dataset} "
            f"with seqlen={args.seqlen}."
        )

    print(
        f"Loaded dataset={args.dataset}, total_tokens={total_tokens}, "
        f"eval_windows={len(windows)}, window_size={args.seqlen}"
    )

    llm, sampling_params = build_llm(args)

    total_nll = 0.0
    total_predicted_tokens = 0
    eval_start = time.perf_counter()

    for batch in chunk_list(windows, args.batch_size):
        vllm_inputs = [{"prompt_token_ids": prompt_token_ids} for prompt_token_ids in batch]
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
        batch_nll, batch_predicted_tokens = compute_batch_nll(outputs, batch)
        total_nll += batch_nll
        total_predicted_tokens += batch_predicted_tokens

    eval_seconds = time.perf_counter() - eval_start
    ppl = math.exp(total_nll / total_predicted_tokens)

    prompt_tokens = len(windows) * args.seqlen
    tok_per_s = prompt_tokens / eval_seconds if eval_seconds > 0 else float("inf")

    # A detached vLLM worker may close inherited stdout before this parent
    # process prints its summary. Keep an opt-in machine-readable result;
    # do not create a workspace-local log during ordinary evaluation.
    result_path = os.getenv("PPL_RESULT_PATH")
    if result_path:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ppl": ppl,
                    "predicted_tokens": total_predicted_tokens,
                    "eval_seconds": eval_seconds,
                    "throughput_tok_s": tok_per_s,
                    "prompt_tokens": prompt_tokens,
                },
                f,
                sort_keys=True,
            )
            f.write("\n")

    print(f"PPL: {ppl:.6f}")
    print(f"Predicted tokens: {total_predicted_tokens}")
    print(f"Eval elapsed: {eval_seconds:.3f}s")
    print(f"Eval throughput: {tok_per_s:.2f} tok/s")


def main():
    parser = build_parser()
    args = parser.parse_args()
    evaluate_perplexity(args)


if __name__ == "__main__":
    main()
