"""API local: reutiliza a inferência e a validação da v0.5."""
import json
from pathlib import Path
from functools import lru_cache
from threading import Lock
from typing import Annotated
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .train import carregar_modelo
from .inference import gerar_chunk_ia
from .maze import caminho_valido

app = FastAPI(title='Labirintos Infinitos', version='1.0')
CHECKPOINT = Path('models/best.pt')
lock = Lock()
Linha = Annotated[int, Field(strict=True, ge=0, le=9)]
Token = Annotated[int, Field(strict=True, ge=1, le=15)]

class Chunk(BaseModel):
    entrada: Linha
    saida: Linha
    grade: list[list[Token]]
    tentativas: int = Field(ge=1)

class Pedido(BaseModel):
    entrada: Linha = 4
    quantidade: int = Field(default=3, ge=1, le=10)
    temperatura: float = Field(default=1, ge=0.1, le=3)

class Expansao(BaseModel):
    chunks: list[Chunk] = Field(min_length=1, max_length=49)
    quantidade: int = Field(default=1, ge=1, le=10)
    temperatura: float = Field(default=1, ge=0.1, le=3)

@lru_cache(maxsize=1)
def modelo():
    if not CHECKPOINT.is_file():
        raise HTTPException(503, 'Checkpoint ausente. Consulte o README.')
    return carregar_modelo(str(CHECKPOINT))

@app.get('/health')
def health():
    modelo()
    return {'status': 'ok', 'modelo': 'carregado'}

@app.get('/model/info')
def info():
    m = modelo()
    return {'arquitetura': 'MiniGPT causal', 'parametros': sum(p.numel() for p in m.parameters()), 'dispositivo': str(next(m.parameters()).device), 'checkpoint': str(CHECKPOINT), 'chunk': [10,10], 'vocabulario': 26}

@app.get('/metrics')
def metrics():
    return {p.stem: json.loads(p.read_text(encoding='utf-8')) for p in [Path('reports/metricas.json'), Path('reports/experimentos.json')] if p.is_file()}

def novos(entrada, quantidade, temperatura):
    resultado = []
    with lock:
        m = modelo()
        for _ in range(quantidade):
            try:
                grade, saida, tentativas = gerar_chunk_ia(m, entrada, temperature=temperatura)
            except RuntimeError as exc:
                raise HTTPException(503, str(exc)) from exc
            resultado.append(Chunk(entrada=entrada, saida=saida, grade=grade.tolist(), tentativas=tentativas))
            entrada = saida
    return resultado

def resposta(chunks):
    return {'chunks': chunks, 'quantidade': len(chunks), 'entrada': chunks[0].entrada, 'saida': chunks[-1].saida}

@app.post('/generate')
def generate(p: Pedido):
    return resposta(novos(p.entrada, p.quantidade, p.temperatura))

@app.post('/expand')
def expand(p: Expansao):
    if len(p.chunks) + p.quantidade > 50:
        raise HTTPException(422, 'Limite de 50 chunks por mapa nesta demonstração.')
    for i, c in enumerate(p.chunks):
        if len(c.grade) != 10 or any(len(r) != 10 for r in c.grade) or not caminho_valido(np.array(c.grade), c.entrada, c.saida):
            raise HTTPException(422, 'Chunk inválido.')
        if i and p.chunks[i-1].saida != c.entrada:
            raise HTTPException(422, 'Conexão inválida.')
    return resposta(p.chunks + novos(p.chunks[-1].saida, p.quantidade, p.temperatura))
