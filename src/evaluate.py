import os, json, math, time, random, hashlib
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .maze import *
from .tokenizer import VOCAB_SIZE
from .dataset import carregar, ler
from .inference import gerar_chunk_ia
from .baseline import gerar_comparacao
from .visualization import desenhar_chunks
SEED=42
@torch.inference_mode()
def avaliar_teste(model, test_seqs, batch_size=32):
    device=next(model.parameters()).device
    model.eval()

    loss_total = 0
    acertos = 0
    total_tokens = 0

    for inicio in range(0, len(test_seqs), batch_size):

        batch = torch.as_tensor(
            test_seqs[inicio:inicio + batch_size],
            dtype=torch.long,
            device=device
        )

        X = batch[:, :-1]
        Y = batch[:, 1:]

        logits = model(X)

        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            Y.reshape(-1),
            reduction="sum"
        )

        loss_total += loss.item()

        previsoes = logits.argmax(dim=-1)

        acertos += (previsoes == Y).sum().item()

        total_tokens += Y.numel()

    loss_media = loss_total / total_tokens
    perplexidade = math.exp(loss_media)
    accuracy = acertos / total_tokens * 100

    resultados = {
        "test_loss": loss_media,
        "perplexidade": perplexidade,
        "accuracy_percentual": accuracy,
        "labirintos_avaliados": len(test_seqs),
        "tokens_avaliados": total_tokens
    }

    for chave, valor in resultados.items():
        print(f"{chave}: {valor:.4f}")

    return resultados

def avaliar(model, quantidade=100):
    os.makedirs("reports", exist_ok=True)
    resultados_teste=avaliar_teste(model, carregar()["test"])
    import time
    import pandas as pd
    import hashlib

    QUANTIDADE_TESTES = quantidade

    chunks_teste = []
    entradas_teste = []
    saidas_teste = []

    registros = []

    entrada = 4

    tempo_inicio = time.perf_counter()

    for i in range(QUANTIDADE_TESTES):

        inicio = time.perf_counter()

        try:
            grid, saida, tentativas = gerar_chunk_ia(
                model,
                entrada,
                max_tentativas=50,
                temperature=1.0
            )

            valido = caminho_valido(
                grid,
                entrada,
                saida
            )

            if not valido:
                raise RuntimeError("Chunk estruturalmente inválido")

            identificador = hashlib.sha256(
                np.asarray(grid, dtype=np.uint8).tobytes()
            ).hexdigest()

            tempo = time.perf_counter() - inicio

            registros.append({
                "teste": i + 1,
                "entrada": entrada,
                "saida": saida,
                "valido": True,
                "tentativas": tentativas,
                "tempo_segundos": tempo,
                "hash": identificador,
                "erro": None
            })

            chunks_teste.append(grid.copy())
            entradas_teste.append(entrada)
            saidas_teste.append(saida)

            print(
                f"Teste {i + 1}/{QUANTIDADE_TESTES} | "
                f"{entrada} -> {saida} | "
                f"Tentativas: {tentativas} | "
                f"Tempo: {tempo:.2f}s | VÁLIDO"
            )

            # Saída anterior torna-se a entrada seguinte
            entrada = saida

        except RuntimeError as erro:

            registros.append({
                "teste": i + 1,
                "entrada": entrada,
                "saida": None,
                "valido": False,
                "tentativas": None,
                "tempo_segundos": time.perf_counter() - inicio,
                "hash": None,
                "erro": str(erro)
            })

            print(f"Teste {i + 1}: FALHOU - {erro}")

    tempo_total = time.perf_counter() - tempo_inicio

    df_testes = pd.DataFrame(registros)

    print("\nTestes finalizados!")
    print(f"Tempo total: {tempo_total:.2f} segundos")


    total = len(df_testes)

    validos = df_testes[df_testes["valido"]]

    sucessos = len(validos)
    falhas = total - sucessos

    taxa_sucesso = sucessos / total * 100

    unicos = validos["hash"].nunique()

    diversidade = (
        unicos / sucessos * 100
        if sucessos else 0
    )

    tentativas_medias = validos["tentativas"].mean()

    tempo_medio = df_testes["tempo_segundos"].mean()

    conexoes_corretas = sum(
        saidas_teste[i] == entradas_teste[i + 1]
        for i in range(len(chunks_teste) - 1)
    )

    total_conexoes = max(0, len(chunks_teste) - 1)

    print("=" * 50)
    print("LABIRINTOS INFINITOS — RESULTADOS")
    print("=" * 50)

    print(f"Total de testes: {total}")
    print(f"Chunks válidos: {sucessos}")
    print(f"Falhas: {falhas}")
    print(f"Taxa de sucesso: {taxa_sucesso:.2f}%")
    print(f"Chunks únicos: {unicos}")
    print(f"Diversidade: {diversidade:.2f}%")
    print(f"Média de tentativas: {tentativas_medias:.2f}")
    print(f"Tempo médio por solicitação: {tempo_medio:.2f}s")

    print(
        f"Conexões corretas: "
        f"{conexoes_corretas}/{total_conexoes}"
    )

    print(f"Células válidas geradas: {sucessos * 100}")
    print(f"Tempo total: {tempo_total:.2f}s")



    plt.figure(figsize=(6, 4))
    plt.bar(
        ["Válidos", "Falhas"],
        [sucessos, falhas],
        color=["seagreen", "tomato"]
    )
    plt.title("Validade dos labirintos gerados")
    plt.ylabel("Quantidade")
    plt.savefig(f"reports/grafico_{plt.gcf().number}.png"); plt.close()


    if not validos.empty:
        plt.figure(figsize=(10, 4))
        plt.plot(
            validos["teste"],
            validos["tentativas"],
            marker="o"
        )
        plt.title("Tentativas para gerar cada chunk válido")
        plt.xlabel("Teste")
        plt.ylabel("Tentativas")
        plt.grid(alpha=0.3)
        plt.savefig(f"reports/grafico_{plt.gcf().number}.png"); plt.close()


    plt.figure(figsize=(10, 4))
    plt.plot(
        df_testes["teste"],
        df_testes["tempo_segundos"],
        marker="o"
    )
    plt.title("Tempo de geração por solicitação")
    plt.xlabel("Teste")
    plt.ylabel("Segundos")
    plt.grid(alpha=0.3)
    plt.savefig(f"reports/grafico_{plt.gcf().number}.png"); plt.close()


    plt.figure(figsize=(6, 4))
    plt.bar(
        ["Únicos", "Repetidos"],
        [unicos, sucessos - unicos]
    )
    plt.title("Diversidade dos chunks")
    plt.ylabel("Quantidade")
    plt.savefig(f"reports/grafico_{plt.gcf().number}.png"); plt.close()


    quantidade_visualizar = min(7, len(chunks_teste))

    if quantidade_visualizar > 0:

        figura = desenhar_chunks(
            chunks_teste[:quantidade_visualizar],
            entradas_teste[:quantidade_visualizar],
            saidas_teste[:quantidade_visualizar],
            titulo="Labirintos Infinitos — Teste de expansão"
        )
        figura.savefig("reports/expansao.png"); plt.close(figura)

    else:
        print("Nenhum chunk válido foi gerado.")



    os.makedirs("reports", exist_ok=True)

    df_testes.to_csv(
        "reports/testes_geracao.csv",
        index=False
    )

    resumo = {
        "total_testes": total,
        "chunks_validos": sucessos,
        "falhas": falhas,
        "taxa_sucesso": taxa_sucesso,
        "chunks_unicos": unicos,
        "diversidade": diversidade,
        "media_tentativas": (
            float(tentativas_medias) if sucessos else None
        ),
        "tempo_medio_segundos": float(tempo_medio),
        "tempo_total_segundos": tempo_total,
        "conexoes_corretas": conexoes_corretas,
        "total_conexoes": total_conexoes,
        "avaliacao_modelo": resultados_teste
    }

    with open(
        "reports/metricas.json",
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            resumo,
            arquivo,
            indent=4,
            ensure_ascii=False
        )

    with open(
        "reports/chunks_gerados.json",
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump([
            {
                "id": i + 1,
                "entrada": entradas_teste[i],
                "saida": saidas_teste[i],
                "grid": grid.flatten().tolist()
            }
            for i, grid in enumerate(chunks_teste)
        ], arquivo, indent=2)

    print("Resultados salvos com sucesso!")



def comparar(model, quantidade=50):
    os.makedirs("reports", exist_ok=True)
    chunks=ler("data/curated/train.jsonl"); n_train=len(chunks)
    # Mesmo conjunto de entradas e uma tentativa por método
    N_COMPARACAO = quantidade
    entradas = [i%10 for i in range(N_COMPARACAO)]
    chaves_treino = {chave_topologia(np.array(c['grid']).reshape(10,10)) for c in chunks[:n_train]}
    registros_comparacao = []
    amostras_invalidas = []
    model.eval()
    for metodo in ["DFS", "Aleatorio restrito", "MiniGPT livre", "MiniGPT restrito"]:
        rng = random.Random(SEED)
        torch.manual_seed(SEED)
        for i, entrada in enumerate(entradas):
            inicio = time.perf_counter()
            grid = None; saida = None; erro = None
            try:
                grid, saida = gerar_comparacao(metodo, entrada, rng, model)
                valido = saida is not None and caminho_valido(grid, entrada, saida)
                if not valido: erro = "Falha na validacao estrutural"
            except RuntimeError as exc:
                valido = False; erro = str(exc)
            tempo = time.perf_counter() - inicio
            chave = chave_topologia(grid) if valido else None
            registros_comparacao.append(dict(metodo=metodo, teste=i+1, entrada=entrada, saida=saida,
                valido=valido, segundos=tempo, topologia=chave,
                presente_treino=chave in chaves_treino if valido else None,
                solucao=comprimento_solucao(grid,entrada,saida) if valido else None, erro=erro))
            if not valido and sum(a['metodo']==metodo for a in amostras_invalidas)<3:
                amostras_invalidas.append(dict(metodo=metodo, entrada=entrada, saida=saida, erro=erro,
                    grid=grid.tolist() if grid is not None else None))
        print(metodo, "finalizado")
    df_comparacao = pd.DataFrame(registros_comparacao)


    # Resumo e evidências da comparação
    linhas = []
    for metodo, grupo in df_comparacao.groupby('metodo', sort=False):
        ok = grupo[grupo.valido]
        linhas.append(dict(metodo=metodo, testes=len(grupo), validos=len(ok),
            validade_pct=100*len(ok)/len(grupo), tempo_medio=grupo.segundos.mean(),
            diversidade_pct=100*ok.topologia.nunique()/len(ok) if len(ok) else None,
            novos_pct=100*(1-ok.presente_treino.astype(bool).mean()) if len(ok) else None,
            solucao_media=ok.solucao.mean() if len(ok) else None))
    tabela_comparacao = pd.DataFrame(linhas)
    print(tabela_comparacao.to_string(index=False))
    tabela_comparacao.set_index('metodo').validade_pct.plot.bar(rot=20, ylim=(0,105), ylabel='Validade (%)')
    plt.tight_layout(); plt.savefig("reports/comparacao.png"); plt.close()
    os.makedirs('reports', exist_ok=True)
    df_comparacao.to_csv('reports/comparacao_detalhada.csv', index=False)
    tabela_comparacao.to_csv('reports/comparacao_resumo.csv', index=False)
    with open('reports/erros_comparacao.json','w') as f: json.dump(amostras_invalidas,f,indent=2)
    print('Relatórios da comparação salvos.')

