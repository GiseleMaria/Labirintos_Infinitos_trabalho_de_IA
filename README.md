# Versão v0.5
Docker CPU (Python 3.12 dentro do container).
```
docker compose build
docker compose up -d mlflow
docker compose run --rm projeto python main.py dados
docker compose run --rm projeto python main.py gerar --entrada 2 --quantidade 3 --sem-janela
```
Veja reports/geracao.png. MLflow: http://localhost:5000.
DVC: execute na raiz do repositório existente.
```
docker compose run --rm projeto dvc init
docker compose run --rm projeto dvc repro dados
docker compose run --rm projeto dvc repro
docker compose run --rm projeto python main.py avaliar --quantidade 100
docker compose run --rm projeto python main.py comparar --quantidade 50
```
Se Git rejeitar a propriedade do diretório no container, execute `git config --global --add safe.directory /app` no mesmo container usando `sh -c` antes de dvc (ou inicialize com dvc local).
DVC guarda cache local; não há remote configurado. Não use dvc push sem configurar remote.
Treino usa três taxas de aprendizado, validação completa, patience 2 avaliações consecutivas sem melhora, seleção pela menor perda de validação. O teste anterior já foi consultado, portanto não é um teste cego novo.
