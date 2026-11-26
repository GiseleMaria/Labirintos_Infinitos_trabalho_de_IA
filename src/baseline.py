import torch
import torch.nn.functional as F
import numpy as np
from .maze import *
from .inference import candidatos, gerar_chunk_ia
from .tokenizer import COND_OFFSET
@torch.inference_mode()
def gerar_comparacao(metodo, entrada, rng, model):
    device=next(model.parameters()).device
    if metodo == "DFS":
        saida = rng.randrange(10)
        grid = gerar_chunk_10x10(entrada, saida, rng)
    elif metodo == "MiniGPT restrito":
        grid, saida, _ = gerar_chunk_ia(model, entrada, max_tentativas=1)
    else:
        prefix = []
        for _ in range(100):
            if metodo == "Aleatorio restrito":
                allowed = candidatos(prefix, entrada)
                if not allowed:
                    raise RuntimeError("Sem candidato")
                token = rng.choice(allowed)
            else:
                idx = torch.tensor([[COND_OFFSET+entrada]+prefix], device=device)
                token = torch.multinomial(F.softmax(model(idx)[0,-1], dim=-1), 1).item()
            prefix.append(token)
        grid = np.array(prefix).reshape(10,10)
        exits = [r for r in range(10) if grid[r,9] & E]
        saida = exits[0] if len(exits)==1 else None
    return grid, saida

