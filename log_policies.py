import os
import torch

path="stage2_best_ler_0.0009765625_15:49.pt"

checkpoint = torch.load(path, map_location="cpu")

print(type(checkpoint))
print(checkpoint.keys())
print(f'LER: {checkpoint["ler"]}  EP: {checkpoint["episode"]} Noise Scalar: {checkpoint["noise_scalar"]}')
#print(f'LER: {checkpoint["ler"]}  EP: {checkpoint["episode"]} ')
sd = checkpoint["agent"]
print("State dict keys:", list(sd.keys())[:15]) 
