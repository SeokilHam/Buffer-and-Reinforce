import torch
import torch.nn.functional as F
import pdb

projection = torch.load("/mnt/server12_hard3/seokil/Booster/Instruct-dWs.pt")
beaver_orth_projection = torch.load("/mnt/server12_hard3/seokil/Booster/Instruct-Beaver-orth.pt")
adv_orth_projection = torch.load("/mnt/server12_hard3/seokil/Booster/Instruct-Beaver-GSM8K-Proj1.pt")

sum = 0
count = 0
for name, params in projection.items():
    if 'q_proj' in name:
        # P_out, P_in, P_flat = projection[name]
        # P_orth_out, P_orth_in = beaver_orth_projection[name]
        # P_adv_orth_out, P_adv_orth_in = adv_orth_projection[name]
        d_align = projection[name]
        pdb.set_trace()
        # print(name)
        # print(F.cosine_similarity((P_in).view(-1), P_orth_in.view(-1), dim=-1))
        # print(F.cosine_similarity((P_flat).view(-1), P_orth_in.view(-1), dim=-1))
        # print(F.cosine_similarity((torch.eye(P_in.shape[0])-P_in).view(-1), P_orth_in.view(-1), dim=-1))
        # print(F.cosine_similarity((P_orth_out).view(-1), P_adv_orth_out.view(-1), dim=-1))

        # print(F.cosine_similarity((P_out).view(-1), P_orth_out.view(-1), dim=-1))
        # print(F.cosine_similarity((torch.eye(P_out.shape[0])-P_out).view(-1), P_orth_out.view(-1), dim=-1))
        # print(F.cosine_similarity((P_orth_in).view(-1), P_adv_orth_in.view(-1), dim=-1))
        # sum += F.cosine_similarity((torch.eye(P_in.shape[0])-P_in).view(-1), P_in.view(-1), dim=-1)
        count += 1

print("average: ", sum/count)