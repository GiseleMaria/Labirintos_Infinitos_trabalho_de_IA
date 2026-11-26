# Labirintos Infinitos — v0.4

Continuação da v0.3: mesmo DFS, tokens e MiniGPT, separados em módulos. Dataset sintético gerado localmente em batch (não é uma coleção de imagens). RAW registra candidatos; PROCESSED valida e deduplica topologias; CURATED guarda tokens e split 80/10/10. O Parquet contém metadados, o JSONL contém grades e tokens. A ordem e seed anteriores são preservadas para reutilizar o checkpoint.

## Executar no Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py dados
```

Copie `best_perfect_maze_gpt.pt` para `models/best.pt` ou use `--checkpoint caminho.pt`. Se não possuir os pesos, execute `python main.py treinar` no ambiente virtual. Depois `python main.py avaliar`, `python main.py comparar` e `python main.py gerar` (use o executável do ambiente virtual). `--quantidade 10` reduz os pedidos de avaliação; `--sem-parquet` permite gerar apenas JSONL para diagnóstico. Não use dataset reduzido para a avaliação final.

## Dados e 5 Vs
Volume: 1500 exemplos, 150 mil células; velocidade: geração em lote; variedade: JSONL e metadados Parquet; veracidade: validação estrutural e deduplicação antes do split; valor: treinamento e avaliação de geração incremental. Não é processamento distribuído nem streaming. O manifesto registra contagens e hashes. O teste anterior já foi consultado na v0.2, portanto não é um conjunto totalmente inédito para decisões futuras. DVC e MLflow entram na próxima versão.
