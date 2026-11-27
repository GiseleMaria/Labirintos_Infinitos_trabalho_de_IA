import os
import json
import requests
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from src.visualization import desenhar_chunks

URL = os.getenv('API_URL', 'http://localhost:8000')
st.set_page_config(page_title='Labirintos Infinitos', layout='wide')
st.title('Labirintos Infinitos')
st.caption('MiniGPT próprio • chunks 10 × 10 • expansão incremental')
entrada = st.sidebar.slider('Linha de entrada', 0, 9, 4)
quantidade = st.sidebar.slider('Chunks por operação', 1, 10, 3)
temperatura = st.sidebar.slider('Temperatura', 0.1, 3.0, 1.0, 0.1)
if 'mapa' not in st.session_state:
    st.session_state.mapa = None

def executar(endpoint, payload):
    try:
        with st.spinner('Gerando com a rede neural...'):
            r = requests.post(URL + endpoint, json=payload, timeout=600)
            r.raise_for_status()
            st.session_state.mapa = r.json()
    except requests.RequestException as exc:
        st.error(str(exc))

if st.sidebar.button('Gerar novo labirinto'):
    executar('/generate', dict(entrada=entrada, quantidade=quantidade, temperatura=temperatura))
mapa = st.session_state.mapa
if mapa and st.sidebar.button('Expandir labirinto', disabled=mapa['quantidade']+quantidade>50):
    executar('/expand', dict(chunks=mapa['chunks'], quantidade=quantidade, temperatura=temperatura))
mapa = st.session_state.mapa
if mapa:
    a,b,c = st.columns(3)
    a.metric('Chunks', mapa['quantidade']); b.metric('Entrada', mapa['entrada']); c.metric('Saída', mapa['saida'])
    chunks = mapa['chunks']
    # Exibe grupos para evitar figuras excessivamente largas.
    for i in range(0,len(chunks),5):
        bloco = chunks[i:i+5]
        fig = desenhar_chunks([np.array(x['grade']) for x in bloco], [x['entrada'] for x in bloco], [x['saida'] for x in bloco], f'Chunks {i+1}–{i+len(bloco)}')
        st.pyplot(fig); plt.close(fig)
    st.download_button('Exportar JSON', json.dumps(mapa,indent=2), 'labirinto.json', 'application/json')
else:
    st.info('Escolha os parâmetros e gere o primeiro labirinto.')
with st.expander('Modelo e métricas'):
    try:
        for endpoint in ['/model/info','/metrics']:
            r=requests.get(URL+endpoint,timeout=15); r.raise_for_status(); st.json(r.json())
    except requests.RequestException as exc:
        st.warning(str(exc))
