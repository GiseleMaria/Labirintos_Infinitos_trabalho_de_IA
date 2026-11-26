import matplotlib.pyplot as plt
from .maze import N,E,S,W
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
    return fig

