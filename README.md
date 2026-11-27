# Labirintos Infinitos

Projeto integrador de Redes Neurais e Deep Learning, Big Data e Engenharia de Dados e MLOps. Um MiniGPT causal criado em PyTorch aprende a gerar labirintos ramificados de 10 × 10 células. Cada novo chunk começa na saída do anterior. Não utiliza Ollama, modelos pré-treinados ou serviços externos de IA.

## 1. Como executar (Windows / PowerShell)

Instale Git e Docker Desktop com WSL 2. Abra o Docker Desktop e aguarde a inicialização. No terminal:

```powershell
git clone https://github.com/GiseleMaria/Labirintos_Infinitos_trabalho_de_IA.git
cd Labirintos_Infinitos_trabalho_de_IA
docker compose up -d --build mlflow
```

O primeiro build pode demorar. Não é necessário instalar Python no Windows. Se já possui o projeto, use sua pasta existente em vez de clonar novamente.

### Obter os pesos e dados

Os pesos e datasets não estão diretamente no GitHub. Se recebeu o backup da equipe, copie `models/`, `data/`, `reports/` e, opcionalmente, `mlflow-data/` e `.dvc/cache/` para a pasta correspondente. Não substitua pesos já treinados sem backup. Se recebeu o cache DVC correspondente ao commit, `docker compose run --rm projeto dvc checkout` pode restaurar suas saídas. Não há remoto DVC configurado por padrão.

Se não recebeu os pesos, reproduza o treinamento do zero:

```powershell
docker compose run --rm projeto dvc repro dados
docker compose run --rm projeto dvc repro experimentos
```

Caso esteja usando uma cópia sem `.dvc/`, execute primeiro `docker compose run --rm projeto dvc init`. Os três treinamentos podem demorar; não os execute novamente se já possui o checkpoint selecionado. O MLflow deve estar ativo. A geração do dataset usa seed 42.

### Abrir a aplicação

```powershell
docker compose up -d api interface
docker compose ps
```

- Interface: http://localhost:8501
- API Swagger: http://localhost:8000/docs
- MLflow: http://localhost:5000 (selecione Model training)

Na interface, escolha a entrada (0 a 9), quantidade e temperatura. Clique em Gerar novo labirinto e depois Expandir labirinto. O mapa permanece na sessão do navegador e pode ser exportado em JSON. O limite de 50 chunks por mapa protege os recursos da demonstração; o algoritmo não depende de um tamanho global fixo. Não é um serviço de produção.

## 2. API

| Método | Endpoint | Uso |
|---|---|---|
| GET | `/health` | Confirma carregamento do checkpoint |
| GET | `/model/info` | Arquitetura, parâmetros e dispositivo |
| GET | `/metrics` | Relatórios locais disponíveis |
| POST | `/generate` | Geração inicial |
| POST | `/expand` | Adiciona chunks sem alterar os anteriores |

No PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/health
$body = @{entrada=5; quantidade=2; temperatura=1.0} | ConvertTo-Json
$mapa = Invoke-RestMethod http://localhost:8000/generate -Method Post -ContentType 'application/json' -Body $body
$exp = @{chunks=$mapa.chunks; quantidade=1; temperatura=1.0} | ConvertTo-Json -Depth 20
$mapa = Invoke-RestMethod http://localhost:8000/expand -Method Post -ContentType 'application/json' -Body $exp
$mapa.quantidade
```

Resultado esperado: 3 chunks. A API rejeita entradas fora dos limites e mapas inválidos. Se a geração falhar, responde com erro, sem substituir o mapa anterior.

## 3. Engenharia de Dados

Pipeline `python main.py dados`: RAW (JSONL sintético via DFS próprio) → PROCESSED (validação e deduplicação topológica) → CURATED (train/val/test e metadados Parquet). São 1.500 chunks, divididos em 1.200/150/150; seed 42; manifest com hashes SHA256. Verifique os números reais no manifest. A divisão ocorre após deduplicação. Cada registro tem 101 tokens: uma condição de entrada e 100 células. Vocabulário: 26 tokens.

5 Vs: volume pequeno e controlado (não é alegado um dataset massivo); variedade em topologias e formatos JSONL/Parquet; velocidade representada pelo processamento e geração incremental; veracidade por validação e deduplicação; valor pela geração de mapas conectados.

```powershell
docker compose run --rm projeto dvc dag
docker compose run --rm projeto dvc status
Get-Content data/manifest.json
```

## 4. Rede neural e experimentos

MiniGPT causal: embedding 64, 2 blocos Transformer, 4 cabeças, contexto 128, FFN 256, 111.258 parâmetros. Treinado do zero por previsão do próximo token e cross-entropy, AdamW. N=1, E=2, S=4, W=8 codificam as aberturas. A inferência usa logits da rede e restringe candidatos por conectividade, reciprocidade de paredes e ausência de ciclos. A validade estrutural não é atribuída exclusivamente à aprendizagem neural. Não existe fallback DFS na inferência.

Experimentos observados na v0.5 (seed 42, batch 32, máximo 2.000 steps, avaliação a cada 250, patience 2):

| Learning rate | Melhor val loss | Melhor step | Final |
|---|---:|---:|---:|
| 0.001 | 0.804223 | 1000 | 1500 (Early Stopping) |
| 0.0003 | 0.840474 | 2000 | 2000 |
| 0.0001 | 1.287111 | 2000 | 2000 |

O menor loss de validação selecionou o checkpoint de LR 0.001. Após 1000 steps a perda de treino continuou caindo enquanto a de validação subiu, motivando a parada. Os demais não demonstraram esse comportamento no intervalo observado. Consulte os runs reais e `reports/experimentos.json`. O conjunto de teste histórico já foi utilizado em versões anteriores; não é uma avaliação independente de todas as decisões do projeto.

## 5. Avaliação e testes

```powershell
docker compose run --rm projeto python -m pytest -q
docker compose run --rm projeto python main.py gerar --entrada 5 --quantidade 3 --sem-janela
docker compose run --rm projeto python main.py avaliar --quantidade 100
docker compose run --rm projeto python main.py comparar --quantidade 50
```

Os testes HTTP simulados isolam o contrato da API; há também um teste com checkpoint real (ignorado se o arquivo não existir). A avaliação mede perda, perplexidade, acurácia de tokens e geração; a comparação inclui DFS, aleatório restrito, MiniGPT livre e MiniGPT restrito. Consulte os valores reais nos relatórios `reports/metricas.json` e `reports/comparacao_resumo.csv`. Faça backup dos relatórios antes de executar novamente. As métricas podem variar. A imagem de geração fica em `reports/geracao.png`.

## 6. Estrutura e evolução

- `main.py`: CLI original; `src/maze.py`, `dataset.py`, `tokenizer.py`: dados e regras.
- `src/model.py`, `train.py`, `inference.py`: arquitetura, treinamento e geração.
- `src/evaluate.py`, `visualization.py`: avaliação e gráficos.
- `dvc.yaml`, `dvc.lock`: pipeline; `mlflow-data/`: tracking local.
- `src/api.py`, `interface/app.py`, `tests/`: camada de aplicação adicionada sem substituir a CLI.
- `Dockerfile`, `compose.yaml`, `requirements.txt`: ambiente e dependências.

A API chama a mesma inferência da CLI. A interface chama a API por HTTP. As versões anteriores continuam sendo a base desta atualização.

## 7. Problemas comuns e encerramento

- Checkpoint ausente: copie o backup ou reproduza o treinamento.
- MLflow 403: preserve a lista `--allowed-hosts` do Compose. Use localhost no navegador e `mlflow:5000` dentro do Docker.
- Porta ocupada: pare o serviço que a utiliza ou altere apenas a porta publicada no Compose.
- Logs: `docker compose logs --tail=50 api interface mlflow`.
- Finalizar: `docker compose stop`. Dados e pesos persistem nas pastas locais.
- Backup: pare os serviços e copie o projeto incluindo `.git`, `.dvc/cache`, datasets, modelos, relatórios e `mlflow-data`. Não compartilhe credenciais. Não é necessário copiar `.venv`.

Os serviços são locais e as portas são vinculadas a 127.0.0.1. As dependências usam intervalos de versão; o ambiente não é um lockfile integral. Para reprodução estrita registre as versões efetivamente instaladas (`docker compose run --rm projeto pip freeze`).
