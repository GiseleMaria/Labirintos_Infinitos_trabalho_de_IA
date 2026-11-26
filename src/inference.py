import torch
import torch.nn.functional as F
import numpy as np
from .maze import *
from .tokenizer import COND_OFFSET
def candidatos(prefix, in_row):
    i = len(prefix); r,c = divmod(i,10)
    used_exit = any(v & E for j,v in enumerate(prefix) if j%10 == 9)
    result = []
    for v in range(1,16):
        if bool(v & N) != (r > 0 and bool(prefix[i-10] & S)): continue
        if bool(v & W) != (bool(prefix[i-1] & E) if c else r == in_row): continue
        if r == 9 and v & S: continue
        if c == 9 and ((used_exit and v & E) or (r == 9 and not used_exit and not v & E)): continue
        # União de componentes incluindo as células futuras que já possuem uma conexão.
        parent = list(range(100)); active = set(range(i+1)); cycle = False
        def root(a):
            while parent[a] != a:
                a = parent[a]
            return a
        for j,t in enumerate(prefix+[v]):
            for bit,k,allowed in [(E,j+1,j%10<9),(S,j+10,j//10<9)]:
                if allowed and t & bit:
                    active.add(k); a,b = root(j),root(k)
                    if a == b: cycle = True; break
                    parent[a] = b
            if cycle: break
        if cycle: continue
        roots = {root(j) for j in active}
        frontier = {root(j) for j in active if j>i}
        # Um componente não pode ficar isolado das células ainda não geradas.
        if i < 99 and (roots-frontier): continue
        if i == 99 and len(roots)!=1: continue
        result.append(v)
    return result

@torch.inference_mode()
def gerar_chunk_ia(model, in_row, max_tentativas=50, temperature=1.0):
    if not 0 <= in_row < 10 or temperature <= 0: raise ValueError('Entrada ou temperatura inválida')
    device = next(model.parameters()).device
    model.eval()
    for tentativa in range(1,max_tentativas+1):
        prefix=[]
        for _ in range(100):
            allowed=candidatos(prefix,in_row)
            if not allowed: break
            idx=torch.tensor([[COND_OFFSET+in_row]+prefix],dtype=torch.long,device=device)
            logits=model(idx)[0,-1,allowed]/temperature
            token=allowed[torch.multinomial(F.softmax(logits,dim=-1),1).item()]
            prefix.append(token)
        if len(prefix)!=100: continue
        grid=np.array(prefix,dtype=np.int64).reshape(10,10)
        exits=[r for r in range(10) if grid[r,9]&E]
        if len(exits)==1 and caminho_valido(grid,in_row,exits[0]):
            return grid,exits[0],tentativa
    raise RuntimeError('Nenhum chunk válido no limite de tentativas; mapa não alterado.')

