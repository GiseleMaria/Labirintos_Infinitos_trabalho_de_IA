# Labirintos Infinitos — v01
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

    # Abre algumas conexões extras para formar ciclos/múltiplos caminhos (multicursal)
    for _ in range(18):
        r = rng.randrange(rows)
        c = rng.randrange(cols)
        bit, dr, dc = rng.choice(DIRS)
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            grid[r, c] |= bit
            grid[nr, nc] |= OPP[bit]

    # Marca abertura externa de entrada (Oeste na linha in_row) e saída (Leste na linha out_row)
    grid[in_row, 0] |= W
    grid[out_row, cols - 1] |= E
    return grid

def caminho_valido(grid: np.ndarray, in_row: int, out_row: int) -> bool:
    rows, cols = grid.shape
    if not (grid[in_row, 0] & W) or not (grid[out_row, cols - 1] & E):
        return False
    q = deque([(in_row, 0)])
    vist = {(in_row, 0)}
    while q:
        r, c = q.popleft()
        if (r, c) == (out_row, cols - 1):
            return True
        for bit, dr, dc in DIRS:
            if grid[r, c] & bit:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in vist:
                    vist.add((nr, nc))
                    q.append((nr, nc))
    return False

rng = random.Random(SEED)
chunks = []
for _ in range(1500):  # comece com ~1.500 no Colab e escale para 10.000+ no pipeline local
    in_r = rng.randrange(10)
    out_r = rng.randrange(10)
    g = gerar_chunk_10x10(in_r, out_r, rng)
    if caminho_valido(g, in_r, out_r):
        chunks.append({"in_row": in_r, "out_row": out_r, "grid": g.flatten().tolist()})

print("Chunks válidos gerados:", len(chunks))

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
    src = train_seqs if split == "train" else val_seqs
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
            torch.save({"model_state": model.state_dict(), "best_val": best_val}, "best_maze_gpt.pt")

print("Melhor val_loss:", round(best_val, 4))

# %% Célula 6
# Célula 6 — Geração autoregressiva de um chunk 10x10 conectado ao chunk anterior
@torch.inference_mode()
def gerar_chunk_ia(model, in_row: int, out_row: int, max_tentativas: int = 50):
    model.eval()
    for _ in range(max_tentativas):
        idx = torch.tensor([[COND_OFFSET + in_row]], dtype=torch.long, device=device)
        # Gera exatamente 100 células (10x10)
        for _ in range(100):
            logits = model(idx[:, -model.max_T :])
            probs = F.softmax(logits[:, -1, :16], dim=-1)  # restringe a bitmasks 0..15
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)

        grid = idx[0, 1:].cpu().numpy().reshape(10, 10)
        # Garante abertura na saída solicitada para encadear o próximo chunk
        grid[in_row, 0] |= W
        grid[out_row, 9] |= E
        if caminho_valido(grid, in_row, out_row):
            return grid, True
    return grid, False

ckpt = torch.load("best_maze_gpt.pt", map_location=device)
model.load_state_dict(ckpt["model_state"])

# Exemplo de encadeamento: Chunk A sai na linha 4 -> Chunk B entra na linha 4 e sai na linha 7
chunk_a, ok_a = gerar_chunk_ia(model, in_row=2, out_row=4)
chunk_b, ok_b = gerar_chunk_ia(model, in_row=4, out_row=7)
print("Chunk A válido:", ok_a, "| shape:", chunk_a.shape)
print("Chunk B válido:", ok_b, "| entrada == saída A:", bool(chunk_b[4, 0] & W) and bool(chunk_a[4, 9] & E))

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


# Gera 3 chunks conectados: A (2 -> 4), B (4 -> 7), C (7 -> 3)
conexoes = [(4, 5), (5, 9), (9, 9)]
chunks_gerados = []
in_lista, out_lista = [], []

for r_in, r_out in conexoes:
    ch, ok = gerar_chunk_ia(model, in_row=r_in, out_row=r_out)
    chunks_gerados.append(ch)
    in_lista.append(r_in)
    out_lista.append(r_out)
    print(f"Chunk ({r_in} -> {r_out}) | válido: {ok}")

desenhar_chunks(chunks_gerados, in_lista, out_lista, titulo="Mapa contínuo: 3 chunks (30x10)")
