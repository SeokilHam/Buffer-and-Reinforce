import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import load_file as load_safetensors_file


EPS = 1e-12
LORA_A_PATTERN = re.compile(r"^(.*)\.lora_A(?:\.[^.]+)?\.weight$")
LORA_B_PATTERN = re.compile(r"^(.*)\.lora_B(?:\.[^.]+)?\.weight$")
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Layer-wise analysis for SafetyLoRA, SafeLoRA, and UtilityLoRA."
    )
    parser.add_argument(
        "--safetylora",
        default="/mnt/server12_hard3/seokil/Booster/ckpt/gsm8k/Meta-Llama-3-8B-Instruct_Jailbroken_Safe_LAT5000_cosine_1e-5_3/",
    )
    parser.add_argument(
        "--safelora",
        default="/mnt/server12_hard3/seokil/Booster/ckpt/gsm8k/Meta-Llama-3-8B-Instruct_Jailbroken_LAT5000_cosine_1e-5_3/",
    )
    parser.add_argument(
        "--utilitylora",
        default="/mnt/server12_hard3/seokil/Booster/ckpt/gsm8k/Meta-Llama-3-8B-Instruct_Jaillbroken_user_LAT5000_cosine_regul0_0.1_1000_1e-5_3/",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--rank_eps", type=float, default=0.01)
    parser.add_argument("--safelora_projection_mode", choices=["onto", "remove"], default="onto")
    parser.add_argument("--top_k", type=int, default=10, help="How many top/bottom layers to print")
    parser.add_argument(
        "--output_json",
        default="/mnt/server12_hard3/seokil/Booster/layerwise_safelora_analysis.json",
    )
    parser.add_argument(
        "--output_csv",
        default="/mnt/server12_hard3/seokil/Booster/layerwise_safelora_analysis.csv",
    )
    return parser.parse_args()


def resolve_state_file(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_file():
        lowered = path.name.lower()
        if "projection" in lowered or "project_matrix" in lowered:
            raise ValueError(
                f"'{path}' looks like a projection matrix file, not a weight-delta checkpoint."
            )
        return path

    for candidate_name in ("adapter_model.safetensors", "adapter_model.bin", "pytorch_model.bin"):
        candidate = path / candidate_name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find adapter weights under '{path_str}'.")


def load_state(path_str: str) -> Dict[str, torch.Tensor]:
    state_path = resolve_state_file(path_str)
    if state_path.suffix == ".safetensors":
        state = load_safetensors_file(str(state_path), device="cpu")
    else:
        state = torch.load(state_path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Loaded object from '{state_path}' is not a state dict.")
    return state


def canonicalize_layer_key(name: str) -> str:
    name = re.sub(r"\.lora_[AB](?:\.[^.]+)?\.weight$", "", name)
    name = re.sub(r"\.weight$", "", name)
    for prefix in ("base_model.model.model.", "base_model.model.", "model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def contains_lora_keys(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(LORA_A_PATTERN.match(key) or LORA_B_PATTERN.match(key) for key in state_dict.keys())


def extract_tensor_from_dense_value(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item
    return None


def extract_lora_pairs(state_dict: Dict[str, torch.Tensor], label: str) -> Dict[str, Dict[str, torch.Tensor]]:
    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        match_a = LORA_A_PATTERN.match(key)
        if match_a:
            pairs.setdefault(canonicalize_layer_key(match_a.group(1)), {})["A"] = value.detach().cpu().float()
            continue
        match_b = LORA_B_PATTERN.match(key)
        if match_b:
            pairs.setdefault(canonicalize_layer_key(match_b.group(1)), {})["B"] = value.detach().cpu().float()

    missing = [key for key, tensors in pairs.items() if "A" not in tensors or "B" not in tensors]
    if missing:
        raise ValueError(f"{label} has incomplete LoRA pairs.")
    if not pairs:
        raise ValueError(f"No LoRA tensors found in {label}.")
    return pairs


def build_delta_map_from_lora_pairs(pairs: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {key: tensors["B"] @ tensors["A"] for key, tensors in pairs.items()}


def extract_dense_delta_map(state_dict: Dict[str, torch.Tensor], label: str) -> Dict[str, torch.Tensor]:
    dense_map: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        tensor = extract_tensor_from_dense_value(value)
        if tensor is None or tensor.ndim != 2:
            continue
        canonical_key = canonicalize_layer_key(key)
        if not any(module in canonical_key for module in TARGET_MODULES):
            continue
        dense_map[canonical_key] = tensor.detach().cpu().float()
    if not dense_map:
        raise ValueError(f"No dense 2D target-module tensors found in {label}.")
    return dense_map


def load_reference_delta_map(path_str: str, label: str) -> Tuple[Dict[str, torch.Tensor], str]:
    state_dict = load_state(path_str)
    if contains_lora_keys(state_dict):
        return build_delta_map_from_lora_pairs(extract_lora_pairs(state_dict, label)), "lora"
    return extract_dense_delta_map(state_dict, label), "dense"


def cosine_similarity(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> float:
    flat_a = tensor_a.reshape(-1).float()
    flat_b = tensor_b.reshape(-1).float()
    denom = float(torch.norm(flat_a) * torch.norm(flat_b))
    if denom <= EPS:
        return math.nan
    return float(torch.dot(flat_a, flat_b) / denom)


def compatible_keys(map_a: Dict[str, torch.Tensor], map_b: Dict[str, torch.Tensor]) -> List[str]:
    return sorted(
        key
        for key in (set(map_a.keys()) & set(map_b.keys()))
        if tuple(map_a[key].shape) == tuple(map_b[key].shape)
    )


def extract_module_name(layer_key: str) -> str:
    for module in TARGET_MODULES:
        if layer_key.endswith(module):
            return module
    for module in TARGET_MODULES:
        if module in layer_key:
            return module
    return "unknown"


def extract_layer_index(layer_key: str) -> int:
    match = re.search(r"layers\.(\d+)\.", layer_key)
    return int(match.group(1)) if match else -1


def svd2_merge_delta(
    utility_a: torch.Tensor,
    utility_b: torch.Tensor,
    safety_a: torch.Tensor,
    safety_b: torch.Tensor,
    alpha: float,
    eps: float,
) -> torch.Tensor:
    utility_a = utility_a.float()
    utility_b = utility_b.float()
    safety_a = safety_a.float()
    safety_b = safety_b.float()

    delta_safety = safety_b @ safety_a
    delta_utility = utility_b @ utility_a
    gram = utility_a @ utility_a.T
    evals, evecs = torch.linalg.eigh(gram)

    if evals.numel() == 0 or float(evals.max()) <= EPS:
        return delta_utility + delta_safety

    valid_indices = evals > (eps * evals.max())
    if not bool(valid_indices.any()):
        return delta_utility + delta_safety

    if int(valid_indices.sum().item()) == int(safety_b.shape[1]):
        basis = utility_b
    else:
        basis = utility_b @ evecs[:, valid_indices]

    q_matrix, _ = torch.linalg.qr(basis, mode="reduced")
    delta_safety_perp = delta_safety - alpha * (q_matrix @ (q_matrix.T @ delta_safety))
    return delta_utility + delta_safety_perp


def build_svd2_merged_map(
    utility_pairs: Dict[str, Dict[str, torch.Tensor]],
    utility_deltas: Dict[str, torch.Tensor],
    safety_pairs: Dict[str, Dict[str, torch.Tensor]],
    alpha: float,
    eps: float,
) -> Dict[str, torch.Tensor]:
    merged = {}
    for key in sorted(utility_deltas.keys()):
        if key not in safety_pairs:
            merged[key] = utility_deltas[key].clone()
            continue
        utility_a = utility_pairs[key]["A"]
        utility_b = utility_pairs[key]["B"]
        safety_a = safety_pairs[key]["A"]
        safety_b = safety_pairs[key]["B"]
        if utility_b.shape[0] != safety_b.shape[0] or utility_a.shape[1] != safety_a.shape[1]:
            merged[key] = utility_deltas[key].clone()
            continue
        merged[key] = svd2_merge_delta(utility_a, utility_b, safety_a, safety_b, alpha, eps)
    return merged


def directional_projection(utility_delta: torch.Tensor, reference_delta: torch.Tensor, mode: str) -> torch.Tensor:
    utility_flat = utility_delta.reshape(-1).float()
    reference_flat = reference_delta.reshape(-1).float()
    reference_norm_sq = float(torch.dot(reference_flat, reference_flat))
    if reference_norm_sq <= EPS:
        return utility_delta.float().clone()
    coeff = float(torch.dot(utility_flat, reference_flat) / reference_norm_sq)
    projected = coeff * reference_delta.float()
    if mode == "onto":
        return projected
    return utility_delta.float() - projected


def build_safelora_projected_map(
    utility_deltas: Dict[str, torch.Tensor],
    safelora_deltas: Dict[str, torch.Tensor],
    mode: str,
) -> Dict[str, torch.Tensor]:
    projected = {}
    for key in sorted(utility_deltas.keys()):
        if key not in safelora_deltas or tuple(utility_deltas[key].shape) != tuple(safelora_deltas[key].shape):
            projected[key] = utility_deltas[key].clone()
            continue
        projected[key] = directional_projection(utility_deltas[key], safelora_deltas[key], mode)
    return projected


def utility_direction_metrics(original: torch.Tensor, transformed: torch.Tensor) -> Tuple[float, float]:
    utility_flat = original.reshape(-1).float()
    transformed_flat = transformed.reshape(-1).float()
    utility_energy = float(torch.dot(utility_flat, utility_flat))
    if utility_energy <= EPS:
        return math.nan, math.nan
    coeff = float(torch.dot(transformed_flat, utility_flat) / utility_energy)
    retained_ratio = coeff ** 2
    damage_ratio = max(0.0, 1.0 - retained_ratio)
    return retained_ratio, damage_ratio


def total_energy_ratio(original: torch.Tensor, transformed: torch.Tensor) -> float:
    original_flat = original.reshape(-1).float()
    transformed_flat = transformed.reshape(-1).float()
    denom = float(torch.dot(original_flat, original_flat))
    if denom <= EPS:
        return math.nan
    return float(torch.dot(transformed_flat, transformed_flat) / denom)


def tensor_distribution_stats(tensor: torch.Tensor, prefix: str) -> Dict[str, float]:
    tensor = tensor.float()
    return {
        f"{prefix}_max": float(torch.max(tensor).item()),
        f"{prefix}_min": float(torch.min(tensor).item()),
        f"{prefix}_mean": float(torch.mean(tensor).item()),
        f"{prefix}_var": float(torch.var(tensor, unbiased=False).item()),
    }


def mean(values: List[float]) -> float:
    values = [value for value in values if not math.isnan(value)]
    return float(sum(values) / len(values)) if values else math.nan


def build_layer_rows(
    safety_deltas: Dict[str, torch.Tensor],
    utility_deltas: Dict[str, torch.Tensor],
    safe_deltas: Dict[str, torch.Tensor],
    merged_utility: Dict[str, torch.Tensor],
    projected_utility: Dict[str, torch.Tensor],
) -> List[Dict[str, object]]:
    keys = sorted(set(safety_deltas) & set(utility_deltas) & set(safe_deltas) & set(merged_utility) & set(projected_utility))
    keys = [
        key for key in keys
        if tuple(safety_deltas[key].shape) == tuple(utility_deltas[key].shape) == tuple(safe_deltas[key].shape)
    ]

    rows = []
    for key in keys:
        safety = safety_deltas[key]
        utility = utility_deltas[key]
        safe = safe_deltas[key]
        merged = merged_utility[key]
        projected = projected_utility[key]

        merged_retained, merged_damage = utility_direction_metrics(utility, merged)
        projected_retained, projected_damage = utility_direction_metrics(utility, projected)
        merged_total_ratio = total_energy_ratio(utility, merged)
        projected_total_ratio = total_energy_ratio(utility, projected)

        row = {
            "layer": key,
            "layer_index": extract_layer_index(key),
            "module": extract_module_name(key),
            "shape": list(safety.shape),
            "safety_norm": float(torch.norm(safety).item()),
            "utility_norm": float(torch.norm(utility).item()),
            "safe_norm": float(torch.norm(safe).item()),
            "merged_utility_norm": float(torch.norm(merged).item()),
            "projected_utility_norm": float(torch.norm(projected).item()),
            "cos_safety_safe": cosine_similarity(safety, safe),
            "cos_safety_utility": cosine_similarity(safety, utility),
            "cos_utility_safe": cosine_similarity(utility, safe),
            "cos_merged_projected": cosine_similarity(merged, projected),
            "merged_utility_retained_ratio": merged_retained,
            "merged_utility_damage_ratio": merged_damage,
            "merged_total_energy_ratio": merged_total_ratio,
            "projected_utility_retained_ratio": projected_retained,
            "projected_utility_damage_ratio": projected_damage,
            "projected_total_energy_ratio": projected_total_ratio,
        }
        row.update(tensor_distribution_stats(safety, "safety"))
        row.update(tensor_distribution_stats(utility, "utility"))
        row.update(tensor_distribution_stats(safe, "safe"))
        row.update(tensor_distribution_stats(merged, "merged_utility"))
        row.update(tensor_distribution_stats(projected, "projected_utility"))
        rows.append(row)
    return rows


def summarize_by_module(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(row["module"], []).append(row)

    summary = {}
    for module, module_rows in grouped.items():
        summary[module] = {
            "layers": len(module_rows),
            "mean_cos_safety_safe": mean([row["cos_safety_safe"] for row in module_rows]),
            "mean_cos_safety_utility": mean([row["cos_safety_utility"] for row in module_rows]),
            "mean_cos_utility_safe": mean([row["cos_utility_safe"] for row in module_rows]),
            "mean_cos_merged_projected": mean([row["cos_merged_projected"] for row in module_rows]),
            "mean_safety_norm": mean([row["safety_norm"] for row in module_rows]),
            "mean_utility_norm": mean([row["utility_norm"] for row in module_rows]),
            "mean_safe_norm": mean([row["safe_norm"] for row in module_rows]),
            "mean_merged_damage": mean([row["merged_utility_damage_ratio"] for row in module_rows]),
            "mean_projected_damage": mean([row["projected_utility_damage_ratio"] for row in module_rows]),
        }
    return dict(sorted(summary.items()))


def top_rows(rows: List[Dict[str, object]], key: str, descending: bool, top_k: int) -> List[Dict[str, object]]:
    valid_rows = [row for row in rows if not math.isnan(row[key])]
    return sorted(valid_rows, key=lambda row: row[key], reverse=descending)[:top_k]


def print_top_section(title: str, rows: List[Dict[str, object]], metric: str) -> None:
    print(title)
    for row in rows:
        print(
            f"  {row['layer']} | {metric}={row[metric]:.6f} | "
            f"module={row['module']} | layer_index={row['layer_index']}"
        )
    print()


def save_csv(rows: List[Dict[str, object]], path_str: str) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(payload: Dict[str, object], path_str: str) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main() -> None:
    args = parse_args()

    safety_pairs = extract_lora_pairs(load_state(args.safetylora), "SafetyLoRA")
    utility_pairs = extract_lora_pairs(load_state(args.utilitylora), "UtilityLoRA")
    safety_deltas = build_delta_map_from_lora_pairs(safety_pairs)
    utility_deltas = build_delta_map_from_lora_pairs(utility_pairs)
    safe_deltas, safe_source_type = load_reference_delta_map(args.safelora, "SafeLoRA")

    merged_utility = build_svd2_merged_map(
        utility_pairs=utility_pairs,
        utility_deltas=utility_deltas,
        safety_pairs=safety_pairs,
        alpha=args.alpha,
        eps=args.rank_eps,
    )
    projected_utility = build_safelora_projected_map(
        utility_deltas=utility_deltas,
        safelora_deltas=safe_deltas,
        mode=args.safelora_projection_mode,
    )

    rows = build_layer_rows(
        safety_deltas=safety_deltas,
        utility_deltas=utility_deltas,
        safe_deltas=safe_deltas,
        merged_utility=merged_utility,
        projected_utility=projected_utility,
    )
    module_summary = summarize_by_module(rows)

    payload = {
        "meta": {
            "safetylora": os.path.abspath(args.safetylora),
            "safelora": os.path.abspath(args.safelora),
            "utilitylora": os.path.abspath(args.utilitylora),
            "safelora_source_type": safe_source_type,
            "safelora_projection_mode": args.safelora_projection_mode,
            "svd2_alpha": args.alpha,
            "svd2_rank_eps": args.rank_eps,
            "layers_analyzed": len(rows),
        },
        "module_summary": module_summary,
        "top_alignment": {
            "safety_vs_safe": top_rows(rows, "cos_safety_safe", True, args.top_k),
            "safety_vs_utility": top_rows(rows, "cos_safety_utility", True, args.top_k),
            "merged_vs_projected": top_rows(rows, "cos_merged_projected", True, args.top_k),
        },
        "top_orthogonal": {
            "safety_vs_safe_abs_smallest": top_rows(rows, "cos_safety_safe", False, args.top_k),
            "safety_vs_utility_abs_smallest": top_rows(rows, "cos_safety_utility", False, args.top_k),
            "merged_vs_projected_abs_smallest": top_rows(rows, "cos_merged_projected", False, args.top_k),
        },
        "layer_rows": rows,
    }

    print(f"Layer-wise analysis finished. Layers analyzed: {len(rows)}")
    print(f"SafeLoRA source type: {safe_source_type}")
    print()

    print("Module summary")
    for module, stats in module_summary.items():
        print(
            f"  {module}: layers={stats['layers']} | "
            f"mean cos(safety,safe)={stats['mean_cos_safety_safe']:.6f} | "
            f"mean cos(safety,utility)={stats['mean_cos_safety_utility']:.6f} | "
            f"mean cos(utility,safe)={stats['mean_cos_utility_safe']:.6f}"
        )
    print()

    print_top_section(
        f"Top {args.top_k} aligned layers: SafetyLoRA vs SafeLoRA",
        top_rows(rows, "cos_safety_safe", True, args.top_k),
        "cos_safety_safe",
    )
    print_top_section(
        f"Top {args.top_k} aligned layers: SafetyLoRA vs UtilityLoRA",
        top_rows(rows, "cos_safety_utility", True, args.top_k),
        "cos_safety_utility",
    )
    print_top_section(
        f"Top {args.top_k} orthogonal-looking layers: SafetyLoRA vs SafeLoRA",
        top_rows(rows, "cos_safety_safe", False, args.top_k),
        "cos_safety_safe",
    )
    print_top_section(
        f"Top {args.top_k} orthogonal-looking layers: SafetyLoRA vs UtilityLoRA",
        top_rows(rows, "cos_safety_utility", False, args.top_k),
        "cos_safety_utility",
    )

    save_json(payload, args.output_json)
    save_csv(rows, args.output_csv)
    print(f"Saved JSON report to {args.output_json}")
    print(f"Saved CSV table to {args.output_csv}")


if __name__ == "__main__":
    main()
