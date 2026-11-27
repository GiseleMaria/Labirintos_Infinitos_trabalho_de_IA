import json, random, hashlib
from pathlib import Path
import numpy as np
from .maze import gerar_chunk_10x10, caminho_valido, chave_topologia
from .tokenizer import chunk_to_seq

def salvar(path, registros):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf8', newline='\n') as f:
        for item in registros: f.write(json.dumps(item)+'\n')

def ler(path):
    with Path(path).open(encoding='utf8') as f: return [json.loads(l) for l in f]

def pipeline(total=1500, seed=42, parquet=True):
    if total < 10: raise ValueError('Use pelo menos 10 exemplos')
    raw, processed, seen = [], [], set()
    rng = random.Random(seed)
    invalidos = duplicados = 0
    while len(processed) < total:
        entrada, saida = rng.randrange(10), rng.randrange(10)
        grid = gerar_chunk_10x10(entrada, saida, rng)
        item = dict(id=len(raw)+1, in_row=entrada, out_row=saida, grid=grid.flatten().tolist())
        raw.append(item)
        if not caminho_valido(grid, entrada, saida):
            invalidos += 1; continue
        chave = chave_topologia(grid)
        if chave in seen:
            duplicados += 1; continue
        seen.add(chave)
        processed.append(dict(item, topologia=chave))
    base = Path('data')
    salvar(base/'raw/mazes.jsonl', raw)
    salvar(base/'processed/mazes.jsonl', processed)
    a,b = int(total*.8), int(total*.1)
    splits = {'train':processed[:a], 'val':processed[a:a+b], 'test':processed[a+b:]}
    metadados = []
    for nome, itens in splits.items():
        salvar(base/f'curated/{nome}.jsonl', [dict(i, tokens=chunk_to_seq(i)) for i in itens])
        metadados.extend(dict(id=i['id'], in_row=i['in_row'], out_row=i['out_row'], topologia=i['topologia'], split=nome) for i in itens)
    if parquet:
        import pandas as pd
        pd.DataFrame(metadados).to_parquet(base/'curated/metadata.parquet', index=False)
    manifest = dict(origem='Sintetico - DFS proprio', seed=seed, grade=[10,10], raw=len(raw), processed=len(processed), invalidos=invalidos, duplicados=duplicados, splits={k:len(v) for k,v in splits.items()}, vocabulario=26, tokens_por_chunk=101, parquet=parquet)
    manifest['sha256'] = {f.as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(base.glob('*/*.jsonl'))}
    (base/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps(manifest,indent=2))
    return manifest

def carregar():
    import torch
    return {nome:torch.tensor([i['tokens'] for i in ler(f'data/curated/{nome}.jsonl')],dtype=torch.long) for nome in ('train','val','test')}
