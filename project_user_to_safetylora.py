import argparse
import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoModelForCausalLM, AutoTokenizer


EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True, help="Path to the base model")
    parser.add_argument("--user_lora", required=True, help="Path to the user LoRA adapter directory")
    parser.add_argument("--safety_lora", required=True, help="Path to the safety LoRA adapter directory")
    parser.add_argument("--output_path", required=True, help="Path to save the merged full model")
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Scaling factor applied to the projected user delta",
    )
    parser.add_argument(
        "--projection_mode",
        choices=["onto", "remove"],
        default="onto",
        help=(
            "'onto': keep only the component of UserLoRA along SafetyLoRA. "
            "'remove': remove the component of UserLoRA along SafetyLoRA."
        ),
    )
    parser.add_argument(
        "--cache_dir",
        default="cache",
        help="Transformers cache directory",
    )
    return parser.parse_args()


def resolve_adapter_file(adapter_path: str) -> Path:
    path = Path(adapter_path)
    if path.is_file():
        return path

    for candidate_name in ("adapter_model.safetensors", "adapter_model.bin", "pytorch_model.bin"):
        candidate = path / candidate_name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find adapter weights under '{adapter_path}'. "
        "Expected adapter_model.safetensors or adapter_model.bin."
    )


def load_adapter_state(adapter_path: str):
    state_path = resolve_adapter_file(adapter_path)
    if state_path.suffix == ".safetensors":
        return load_safetensors_file(str(state_path), device="cpu")
    return torch.load(state_path, map_location="cpu")


def check_rank_collapse(user_state, eps=1e-2):
    print("Checking Rank Collapse for UserLoRA")
    total_layers = 0
    collapsed_layers = 0
    avg_effective_rank = 0.0

    print(f"{'Layer Name':<60} | {'Set Rank':<8} | {'Eff Rank':<8} | {'Ratio':<6}")
    print("-" * 95)

    for name, value in user_state.items():
        if "lora_A" not in name:
            continue

        wa = value.float()
        rank = wa.shape[0]
        gram = wa @ wa.T
        evals = torch.linalg.eigvalsh(gram)
        threshold = eps * evals.max()
        eff_rank = int((evals > threshold).sum().item())

        print(f"{name:<60} | {rank:<8} | {eff_rank:<8} | {eff_rank / rank:.2f}")
        total_layers += 1
        avg_effective_rank += eff_rank
        if eff_rank < rank:
            collapsed_layers += 1

    print("-" * 95)
    if total_layers > 0:
        print(f"Summary: {collapsed_layers}/{total_layers} layers have rank collapse.")
        print(f"Average Effective Rank: {avg_effective_rank / total_layers:.2f}")


def project_user_delta_to_safety(wa_u, wb_u, wa_s, wb_s, alpha=1.0, projection_mode="onto"):
    delta_u = wb_u.float() @ wa_u.float()
    delta_s = wb_s.float() @ wa_s.float()

    flat_u = delta_u.reshape(-1)
    flat_s = delta_s.reshape(-1)

    safety_norm_sq = float(torch.dot(flat_s, flat_s))
    if safety_norm_sq <= EPS:
        return delta_u.to(torch.float16)

    coeff = float(torch.dot(flat_u, flat_s) / safety_norm_sq)
    projected = coeff * delta_s

    if projection_mode == "onto":
        final_delta = alpha * projected
    elif projection_mode == "remove":
        final_delta = delta_u - alpha * projected
    else:
        raise ValueError(f"Unsupported projection_mode: {projection_mode}")

    print(
        "UserLoRA Scale: {:.6f} | SafetyLoRA Scale: {:.6f} | Projected User Scale: {:.6f}".format(
            delta_u.norm().item(),
            delta_s.norm().item(),
            final_delta.norm().item(),
        )
    )
    return final_delta.to(torch.float16)


def main():
    args = parse_args()

    print(f"Loading base model from {args.base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        cache_dir=args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, cache_dir=args.cache_dir)

    print("Loading LoRA weights...")
    user_state = load_adapter_state(args.user_lora)
    safety_state = load_adapter_state(args.safety_lora)

    check_rank_collapse(user_state)

    start_time = time.perf_counter()
    print("Projecting UserLoRA onto SafetyLoRA and merging into base model...")

    with torch.no_grad():
        base_params = base_model.state_dict()
        merged_layers = 0
        fallback_layers = 0

        for name in user_state.keys():
            if "lora_A" not in name:
                continue

            base_key = name.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
            if base_key not in base_params:
                continue

            lora_a_name = name
            lora_b_name = name.replace("lora_A", "lora_B")
            ua = user_state[lora_a_name]
            ub = user_state[lora_b_name]

            if lora_a_name in safety_state and lora_b_name in safety_state:
                sa = safety_state[lora_a_name]
                sb = safety_state[lora_b_name]
                delta_w = project_user_delta_to_safety(
                    wa_u=ua,
                    wb_u=ub,
                    wa_s=sa,
                    wb_s=sb,
                    alpha=args.alpha,
                    projection_mode=args.projection_mode,
                )
                merged_layers += 1
            else:
                delta_w = (ub.float() @ ua.float()).to(torch.float16)
                fallback_layers += 1

            base_params[base_key] += delta_w * 2
            print(f"Merged layer: {base_key}")

    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed / 3600:.4f} hours")
    print(f"Projected layers: {merged_layers}")
    print(f"Fallback user-only layers: {fallback_layers}")

    print(f"Saving full model to {args.output_path}...")
    os.makedirs(args.output_path, exist_ok=True)
    base_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print("Done.")


if __name__ == "__main__":
    main()
