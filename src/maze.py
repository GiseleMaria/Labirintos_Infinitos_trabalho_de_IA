import random
import numpy as np
from collections import deque
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

import hashlib
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

