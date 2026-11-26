# Labirintos Infinitos — v02
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
# Inicializa as listas apenas se ainda não existirem
if "chunks_gerados" not in globals():
    chunks_gerados = []

if "in_lista" not in globals():
    in_lista = []

if "out_lista" not in globals():
    out_lista = []


def expandir():
    # Utiliza a saída anterior como próxima entrada
    entrada = out_lista[-1] if out_lista else 4

    # Gera um novo chunk utilizando a IA
    try:
        grid, saida, tentativas = gerar_chunk_ia(
            model,
            entrada
        )

    except RuntimeError as erro:
        print(f"Falha na geração: {erro}")
        return False

    # Armazena o novo chunk válido
    chunks_gerados.append(grid)
    in_lista.append(entrada)
    out_lista.append(saida)

    print(f"Chunk {len(chunks_gerados)} gerado!")
    print(f"Entrada: {entrada}")
    print(f"Saída: {saida}")
    print(f"Tentativas: {tentativas}")
    print("Labirinto válido: True")

    return True

# %% Célula 9
expandir()

desenhar_chunks(
    chunks_gerados,
    in_lista,
    out_lista,
    titulo="Labirintos Infinitos"
)

# %% Célula 10

expandir()

if chunks_gerados:
    desenhar_chunks(
        chunks_gerados,
        in_lista,
        out_lista,
        titulo="Labirintos Infinitos"
    )
