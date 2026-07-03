import torch
import argparse
import json
import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", required=True, help="Path to the original base model (e.g., Llama-3-8b)")
parser.add_argument("--user_lora", required=True)
parser.add_argument("--reinforce_lora", required=True)
parser.add_argument("--output_path", required=True, help="Path to save the fully merged model")
parser.add_argument("--alpha", type=float, default=0.1, help="Scaling factor for reinforce LoRA projection")
parser.add_argument("--scale", type=float, default=0.5, help="Scaling factor for reinforce LoRA projection")

args = parser.parse_args()

print(f"Loading base model from {args.base_model_path}...")
base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model_path, 
    torch_dtype=torch.float16,
    device_map="cpu",
    cache_dir="cache",
)
tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)


print("Loading LoRA weights...")
W_u = torch.load(os.path.join(args.user_lora, "adapter_model.bin"), map_location="cpu")
W_s = torch.load(os.path.join(args.reinforce_lora, "adapter_model.bin"), map_location="cpu")

def load_lora_config(lora_path):
    config_path = os.path.join(lora_path, "adapter_config.json")
    if not os.path.exists(config_path):
        print(f"Warning: adapter_config.json not found in {lora_path}; using LoRA scale 1.0")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_lora_alpha(config, lora_a_name):
    alpha_pattern = config.get("alpha_pattern") or {}
    for pattern, alpha in sorted(alpha_pattern.items(), key=lambda item: len(item[0]), reverse=True):
        if pattern in lora_a_name:
            return float(alpha)
    alpha = config.get("lora_alpha")
    return float(alpha) if alpha is not None else None

def get_lora_scale(config, lora_a_name, lora_a_weight):
    lora_alpha = get_lora_alpha(config, lora_a_name)
    if lora_alpha is None:
        return 1.0
    rank = lora_a_weight.shape[0]
    return lora_alpha / rank

user_lora_config = load_lora_config(args.user_lora)
reinforce_lora_config = load_lora_config(args.reinforce_lora)

def get_orth_projected_delta(Wa_u, Wb_u, Wa_s, Wb_s, alpha=0.1, user_scale=1.0, safety_scale=1.0):
    delta_u = user_scale * (Wb_u.float() @ Wa_u.float())
    delta_s = safety_scale * (Wb_s.float() @ Wa_s.float())
    
    flat_u = delta_u.flatten()
    flat_s = delta_s.flatten()
    
    dot_su = torch.dot(flat_s, flat_u)
    dot_uu = torch.dot(flat_u, flat_u)
    
    proj_coeff = dot_su / (dot_uu + 1e-8)
    delta_s_perp = delta_s - (alpha * proj_coeff * delta_u)
    
    final_delta = delta_u + delta_s_perp
    return final_delta.to(torch.float16) 

def check_rank_collapse(user_lora_path, eps=1e-2):
    print(f"Checking Rank Collapse for: {user_lora_path}")
    W_u = torch.load(os.path.join(user_lora_path, "adapter_model.bin"), map_location="cpu")
    
    total_layers = 0
    collapsed_layers = 0
    avg_effective_rank = 0.0
    
    print(f"{'Layer Name':<50} | {'Set Rank':<8} | {'Eff Rank':<8} | {'Ratio':<6}")
    print("-" * 80)

    for name in W_u.keys():
        if 'lora_A' in name:
            Wa = W_u[name].float()
            r = Wa.shape[0]
 
            G = Wa @ Wa.T
            evals = torch.linalg.eigvalsh(G)

            threshold = eps * evals.max()
            eff_rank = (evals > threshold).sum().item()
            
            print(f"{name:<50} | {r:<8} | {eff_rank:<8} | {eff_rank/r:.2f}")
            
            total_layers += 1
            avg_effective_rank += eff_rank
            if eff_rank < r:
                collapsed_layers += 1

    print("-" * 80)
    print(f"Summary: {collapsed_layers}/{total_layers} layers have rank collapse.")
    print(f"Average Effective Rank: {avg_effective_rank / total_layers:.2f}")

check_rank_collapse(args.user_lora)

def get_fast_qr_projected_delta(Wa_u, Wb_u, Wa_s, Wb_s, alpha=0.1, eps=0.01, user_scale=1.0, safety_scale=1.0):
    delta_s = safety_scale * (Wb_s.float() @ Wa_s.float())

    G = Wa_u.float() @ Wa_u.float().T
    
    evals, evecs = torch.linalg.eigh(G)

    valid_indices = evals > (eps * evals.max())
    
    if not valid_indices.any():
        return user_scale * (Wb_u.float() @ Wa_u.float()) + delta_s


    if valid_indices.sum().item() == Wb_s.shape[1]:
        B_eff = Wb_u.float()
    else:
        V_eff = evecs[:, valid_indices]

        B_eff = Wb_u.float() @ V_eff

    Q, _ = torch.linalg.qr(B_eff, mode='reduced')

    delta_s_perp = delta_s - alpha *  Q @ (Q.T @ delta_s)

    delta_u = user_scale * (Wb_u.float() @ Wa_u.float())
    final_delta = delta_u + delta_s_perp 
    
    print("UserLoRA Scale: ", delta_u.norm().item())
    print("Safety LoRA Projected Scale: ", (delta_s_perp).norm().item())
    print("Final Delta Scale: ", final_delta.norm().item()) 
    
    return final_delta.to(torch.float16)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()

print("Merging LoRAs into Base Model...")
with torch.no_grad():
    base_params = base_model.state_dict()
    processed_layers = set()
    
    for name, param in W_u.items():
        if 'lora_A' in name:
            layer_key = name.replace('.lora_A.weight', '.weight').replace('base_model.model.', '')
            
            base_key = name.replace('base_model.model.', '').replace('.lora_A.weight', '.weight')
            if base_key in base_params:
                lora_a_name = name
                lora_b_name = name.replace('lora_A', 'lora_B')
                
                ua = W_u[lora_a_name]
                ub = W_u[lora_b_name]
                
                if lora_a_name in W_s and lora_b_name in W_s:
                    sa = W_s[lora_a_name]
                    sb = W_s[lora_b_name]
                    user_scale = get_lora_scale(user_lora_config, lora_a_name, ua)
                    safety_scale = get_lora_scale(reinforce_lora_config, lora_a_name, sa)
                    
                    delta_w = get_fast_qr_projected_delta(
                        ua, ub, sa, sb, alpha=args.alpha, user_scale=user_scale, safety_scale=safety_scale
                    )
                else:
                    user_scale = get_lora_scale(user_lora_config, lora_a_name, ua)
                    delta_w = (user_scale * (ub.float() @ ua.float())).to(torch.float16)
                
                base_params[base_key] += delta_w * args.scale
                print(f"Merged layer: {base_key}")
                del delta_w

end_event.record()
torch.cuda.synchronize()
one_shot_time = start_event.elapsed_time(end_event)
print("Estimated one shot time {} (h)".format(one_shot_time/ 1000/3600))
memory_usage = torch.cuda.memory_reserved()
print(f"Memory usage: { memory_usage:.2f} GPU memory used")

print(f"Saving full model to {args.output_path}...")
base_model.save_pretrained(args.output_path)
tokenizer.save_pretrained(args.output_path)
print("Done.")
