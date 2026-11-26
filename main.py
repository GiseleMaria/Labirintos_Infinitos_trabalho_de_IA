# Labirintos Infinitos — v03
# Conversão do notebook para Python com células do VS Code.

# %% Célula 1
# Célula 1 — Ambiente, seed e dispositivo (padrão MiniGPT / PyTorch)
import json
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch:", torch.__version__, "| Device:", device)

# %% Célula 2
# Célula 2 — Geração de chunks 10x10 em bitmask (Norte=1, Leste=2, Sul=4, Oeste=8)
# Garante entrada na borda esquerda (in_row), saída na borda direita (out_row) e caminho válido.
N, E, S, W = 1, 2, 4, 8
DIRS = [(N, -1, 0), (E, 0, 1), (S, 1, 0), (W, 0, -1)]
OPP = {N: S, S: N, E: W, W: E}

def gerar_chunk_10x10(in_row: int, out_row: int, rng: random.Random) -> np.ndarray:
    rows, cols = 10, 10
    grid = np.zeros((rows, cols), dtype=np.int64)
    visited = np.zeros((rows, cols), dtype=bool)

    start = (in_row, 0)
    stack = [start]
    visited[start] = True

    # DFS aleatório para criar corredores, ramificações e becos sem saída
    while stack:
        r, c = stack[-1]
        vizinhos = []
        for bit, dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                vizinhos.append((bit, nr, nc))
        if vizinhos:
            bit, nr, nc = rng.choice(vizinhos)
            grid[r, c] |= bit
            grid[nr, nc] |= OPP[bit]
            visited[nr, nc] = True
            stack.append((nr, nc))
        else:
            stack.pop()

    # Marca abertura externa de entrada (Oeste na linha in_row) e saída (Leste na linha out_row)
    grid[in_row, 0] |= W
    grid[out_row, cols - 1] |= E
    return grid

def caminho_valido(grid, in_row, out_row):
    if grid.shape != (10, 10) or not (0 <= in_row < 10 and 0 <= out_row < 10):
        return False
    if not np.issubdtype(grid.dtype, np.integer) or np.any((grid < 1) | (grid > 15)):
        return False
    edges = 0
    branches = 0
    dead_ends = 0
    for r in range(10):
        for c in range(10):
            degree = 0
            for bit, dr, dc in DIRS:
                nr, nc = r+dr, c+dc
                opened = bool(grid[r,c] & bit)
                if 0 <= nr < 10 and 0 <= nc < 10:
                    if opened != bool(grid[nr,nc] & OPP[bit]):
                        return False
                    degree += int(opened)
                elif opened != ((bit == W and r == in_row) or (bit == E and r == out_row)):
                    return False
            edges += degree
            branches += degree >= 3
            dead_ends += degree == 1 and (r,c) not in [(in_row,0),(out_row,9)]
    if edges != 198 or branches < 2 or dead_ends < 1:
        return False
    seen = {(in_row,0)}
    q = deque(seen)
    while q:
        r,c = q.popleft()
        for bit,dr,dc in DIRS:
            nr,nc = r+dr,c+dc
            if grid[r,c] & bit and 0 <= nr < 10 and 0 <= nc < 10 and (nr,nc) not in seen:
                seen.add((nr,nc)); q.append((nr,nc))
    return len(seen) == 100

rng = random.Random(SEED)
chunks = []
seen_grids = set()
while len(chunks) < 1500:
    in_r, out_r = rng.randrange(10), rng.randrange(10)
    g = gerar_chunk_10x10(in_r, out_r, rng)
    # Deduplicar topologia interna, independentemente das aberturas externas.
    internal = g.copy(); internal[:,0] &= ~W; internal[:,9] &= ~E
    key = tuple(internal.flatten())
    if key not in seen_grids and caminho_valido(g,in_r,out_r):
        seen_grids.add(key)
        chunks.append({'in_row':in_r,'out_row':out_r,'grid':g.flatten().tolist()})
with open('perfect_mazes.jsonl','w',encoding='utf-8') as f:
    for item in chunks: f.write(json.dumps(item)+'\n')
print('Chunks válidos:',len(chunks))

# %% Célula 3
# Célula 3 — Vocabulário de tokens, split (treino/val/teste) e batches
# Tokens 0..15: bitmasks das 100 células
# Tokens 16..25: condição de entrada da linha (16 + in_row), usada no início da sequência
COND_OFFSET = 16
VOCAB_SIZE = 26  # 0..15 (bitmask) + 16..25 (in_row 0..9)
SEQ_LEN = 101    # [COND_OFFSET + in_row] + 100 células (10x10)

def chunk_to_seq(item: dict) -> list[int]:
    return [COND_OFFSET + item["in_row"]] + item["grid"]

seqs = [chunk_to_seq(c) for c in chunks]
n_total = len(seqs)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)

train_seqs = torch.tensor(seqs[:n_train], dtype=torch.long)
val_seqs = torch.tensor(seqs[n_train:n_train + n_val], dtype=torch.long)
test_seqs = torch.tensor(seqs[n_train + n_val:], dtype=torch.long)

def get_batch(split: str, batch_size: int = 32):
    src = {"train": train_seqs, "val": val_seqs, "test": test_seqs}[split]
    idx = torch.randint(0, len(src), (batch_size,))
    batch = src[idx]
    X = batch[:, :-1].to(device)  # shape (B, 100)
    Y = batch[:, 1:].to(device)   # próximo token para cada posição
    return X, Y

Xb, Yb = get_batch("train", 8)
print("X.shape:", Xb.shape, "| Y.shape:", Yb.shape)

# %% Célula 4
# Célula 4 — MiniGPT causal (baseado na estrutura do Hendrick, adaptado a T >= 100)
class Head(nn.Module):
    def __init__(self, d: int, head_dim: int, max_T: int):
        super().__init__()
        self.key = nn.Linear(d, head_dim, bias=False)
        self.query = nn.Linear(d, head_dim, bias=False)
        self.value = nn.Linear(d, head_dim, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(max_T, max_T)))
        self.head_dim = head_dim

    def forward(self, x):
        _, T_cur, _ = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.mask[:T_cur, :T_cur] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return weights @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, d: int, h: int, max_T: int):
        super().__init__()
        assert d % h == 0
        head_dim = d // h
        self.heads = nn.ModuleList([Head(d, head_dim, max_T) for _ in range(h)])
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d: int, h: int, max_T: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, h, max_T)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = FeedForward(d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class MazeMiniGPT(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d=64, max_T=128, L=2, h=4):
        super().__init__()
        self.max_T = max_T
        self.token_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_T, d)
        self.blocks = nn.ModuleList([TransformerBlock(d, h, max_T) for _ in range(L)])
        self.ln_final = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size)

    def forward(self, idx):
        _, T_cur = idx.shape
        pos = torch.arange(T_cur, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.lm_head(x)

model = MazeMiniGPT(vocab_size=VOCAB_SIZE, d=64, max_T=128, L=2, h=4).to(device)
print("Parâmetros treináveis:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# %% Célula 5
# Célula 5 — Treinamento com validação e salvamento do melhor checkpoint
@torch.inference_mode()
def eval_val(model, steps=25, batch_size=32):
    model.eval()
    losses = []
    for _ in range(steps):
        X, Y = get_batch("val", batch_size)
        logits = model(X)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), Y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
best_val = float("inf")
max_steps = 2000  # aumente no Colab/local conforme o tempo disponível

for step in range(1, max_steps + 1):
    X, Y = get_batch("train", batch_size=32)
    logits = model(X)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), Y.reshape(-1))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 250 == 0:
        val_loss = eval_val(model)
        print(f"step {step:4d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "best_val": best_val}, "best_perfect_maze_gpt.pt")

print("Melhor val_loss:", round(best_val, 4))

# %% Célula 6
# Célula 6 — Inferência neural com restrições de árvore
# Cada candidato é avaliado sem alterar os tokens já gerados.
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

ckpt=torch.load('best_perfect_maze_gpt.pt',map_location=device)
model.load_state_dict(ckpt['model_state'])

# %% Célula 7
import matplotlib.pyplot as plt
import numpy as np

def desenhar_chunks(chunks_lista, in_rows, out_rows, titulo="Labirinto (chunks 10x10)"):
    """
    Desenha 1 ou mais chunks (10x10 cada) lado a lado, conectando a borda direita
    de um chunk com a borda esquerda do próximo.
    Bitmask: Norte=1, Leste=2, Sul=4, Oeste=8.
    """
    n_chunks = len(chunks_lista)
    rows, cols = 10, n_chunks * 10

    fig, ax = plt.subplots(figsize=(4 * n_chunks + 1, 4.5))
    ax.set_aspect("equal")
    ax.set_xlim(-0.2, cols + 0.2)
    ax.set_ylim(-0.2, rows + 0.2)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(titulo)

    # Fundo das células
    ax.add_patch(plt.Rectangle((0, 0), cols, rows, facecolor="#0e1b2e", edgecolor="none"))

    for c_idx, grid in enumerate(chunks_lista):
        offset_x = c_idx * 10
        r_in = in_rows[c_idx]
        r_out = out_rows[c_idx]

        # Destaca entrada (verde) e saída (vermelho/laranja) do chunk
        ax.add_patch(plt.Rectangle((offset_x, r_in), 0.35, 1.0, facecolor="#2ecc71", alpha=0.85, edgecolor="none"))
        ax.add_patch(plt.Rectangle((offset_x + 9.65, r_out), 0.35, 1.0, facecolor="#e74c3c", alpha=0.85, edgecolor="none"))

        for r in range(10):
            for col in range(10):
                val = int(grid[r, col])
                x0 = offset_x + col
                y0 = r

                # Norte (y = y0)
                if not (val & N):
                    ax.plot([x0, x0 + 1], [y0, y0], color="#e8f0fb", lw=2)
                # Sul (y = y0 + 1)
                if not (val & S):
                    ax.plot([x0, x0 + 1], [y0 + 1, y0 + 1], color="#e8f0fb", lw=2)
                # Oeste (x = x0) — não desenha parede na entrada externa
                if not (val & W) and not (col == 0 and r == r_in):
                    ax.plot([x0, x0], [y0, y0 + 1], color="#e8f0fb", lw=2)
                # Leste (x = x0 + 1) — não desenha parede na saída externa
                if not (val & E) and not (col == 9 and r == r_out):
                    ax.plot([x0 + 1, x0 + 1], [y0, y0 + 1], color="#e8f0fb", lw=2)

        # Linha divisória sutil entre chunks
        if c_idx > 0:
            ax.axvline(offset_x, color="#78d7ff", linestyle="--", lw=1.2, alpha=0.7)

    plt.tight_layout()
    plt.show()


# A saída prevista pela IA define automaticamente a próxima entrada.
chunks_gerados=[]; in_lista=[]; out_lista=[]
def expandir():
    entrada=out_lista[-1] if out_lista else 4
    grid,saida,tentativas=gerar_chunk_ia(model,entrada)
    chunks_gerados.append(grid); in_lista.append(entrada); out_lista.append(saida)
    print(f'Chunk {len(chunks_gerados)}: {entrada} -> {saida} | tentativas: {tentativas}')
for _ in range(3):
    try: expandir()
    except RuntimeError as exc:
        print(exc); break
if chunks_gerados:
    desenhar_chunks(chunks_gerados,in_lista,out_lista,titulo='Labirintos Infinitos')
# Para continuar, execute expandir() e desenhar_chunks(...) em outra célula.

# %% Célula 8
import math

@torch.inference_mode()
def avaliar_teste(model, batch_size=32):
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


resultados_teste = avaliar_teste(model)

# %% Célula 9
import time
import pandas as pd
import hashlib

QUANTIDADE_TESTES = 100

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

# %% Célula 10
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

# %% Célula 11
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.bar(
    ["Válidos", "Falhas"],
    [sucessos, falhas],
    color=["seagreen", "tomato"]
)
plt.title("Validade dos labirintos gerados")
plt.ylabel("Quantidade")
plt.show()


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
    plt.show()


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
plt.show()


plt.figure(figsize=(6, 4))
plt.bar(
    ["Únicos", "Repetidos"],
    [unicos, sucessos - unicos]
)
plt.title("Diversidade dos chunks")
plt.ylabel("Quantidade")
plt.show()

# %% Célula 12
quantidade_visualizar = min(7, len(chunks_teste))

if quantidade_visualizar > 0:

    desenhar_chunks(
        chunks_teste[:quantidade_visualizar],
        entradas_teste[:quantidade_visualizar],
        saidas_teste[:quantidade_visualizar],
        titulo="Labirintos Infinitos — Teste de expansão"
    )

else:
    print("Nenhum chunk válido foi gerado.")

# %% Célula 13
import json
import os

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

# %% Célula 14
# Avaliação complementar — mesmos candidatos, estratégias diferentes
import time
import hashlib
import pandas as pd

@torch.inference_mode()
def gerar_comparacao(metodo, entrada, rng):
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

def chave_topologia(grid):
    g = grid.copy(); g[:,0] &= ~W; g[:,9] &= ~E
    return hashlib.sha256(g.astype(np.uint8).tobytes()).hexdigest()

def comprimento_solucao(grid, entrada, saida):
    q = deque([(entrada,0,0)])
    vistos = {(entrada,0)}
    while q:
        r,c,d = q.popleft()
        if (r,c)==(saida,9): return d
        for bit,dr,dc in DIRS:
            nr,nc = r+dr,c+dc
            if grid[r,c]&bit and 0<=nr<10 and 0<=nc<10 and (nr,nc) not in vistos:
                vistos.add((nr,nc)); q.append((nr,nc,d+1))

# %% Célula 15
# Mesmo conjunto de entradas e uma tentativa por método
N_COMPARACAO = 50
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
            grid, saida = gerar_comparacao(metodo, entrada, rng)
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

# %% Célula 16
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
plt.tight_layout(); plt.show()
os.makedirs('reports', exist_ok=True)
df_comparacao.to_csv('reports/comparacao_detalhada.csv', index=False)
tabela_comparacao.to_csv('reports/comparacao_resumo.csv', index=False)
with open('reports/erros_comparacao.json','w') as f: json.dump(amostras_invalidas,f,indent=2)
print('Relatórios da comparação salvos.')
