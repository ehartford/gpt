import torch
import torch_directml

from model import BigramLanguageModel, decode

dml = torch_directml.device()
torch.device(dml)

blm = BigramLanguageModel().to(dml)
blm.load_state_dict(torch.load('model.pt', map_location=dml))
context = torch.zeros((1, 1), dtype=torch.long, device=dml)

print(decode(blm.generate(context, max_new_tokens=500)[0].tolist()))
