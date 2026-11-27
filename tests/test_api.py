import random
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from src import api
from src.maze import gerar_chunk_10x10, caminho_valido
import numpy as np

client = TestClient(api.app)

@pytest.fixture
def simulado(monkeypatch):
    # Isola contrato HTTP; não é evidência de inferência neural.
    monkeypatch.setattr(api, 'modelo', lambda: object())
    monkeypatch.setattr(api, 'gerar_chunk_ia', lambda m,e,temperature: (gerar_chunk_10x10(e, (e+1)%10, random.Random(e)), (e+1)%10, 1))

def test_fluxo(simulado):
    assert client.get('/health').status_code == 200
    r=client.post('/generate',json={'entrada':5,'quantidade':2}); assert r.status_code==200
    chunks=r.json()['chunks']; assert chunks[0]['entrada']==5
    r=client.post('/expand',json={'chunks':chunks,'quantidade':2}); assert r.status_code==200
    novos=r.json()['chunks']; assert novos[:2]==chunks
    for a,b in zip(novos,novos[1:]): assert a['saida']==b['entrada']
    for c in novos: assert caminho_valido(np.array(c['grade']),c['entrada'],c['saida'])
    chunks[0]['grade'][0][0]=0
    assert client.post('/expand',json={'chunks':chunks}).status_code==422

@pytest.mark.parametrize('payload',[{'entrada':10},{'entrada':-1},{'quantidade':0},{'temperatura':0}])
def test_entrada_invalida(payload):
    assert client.post('/generate',json=payload).status_code==422

@pytest.mark.skipif(not Path('models/best.pt').exists(),reason='Checkpoint ausente')
def test_inferencia_real():
    import torch
    torch.set_num_threads(2)
    r=client.post('/generate',json={'entrada':7,'quantidade':1}); assert r.status_code==200
    c=r.json()['chunks'][0]; assert caminho_valido(np.array(c['grade']),7,c['saida'])
    r=client.post('/expand',json={'chunks':[c]}); assert r.status_code==200
    assert r.json()['chunks'][0]==c
    assert r.json()['chunks'][1]['entrada']==c['saida']
