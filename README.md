# Labirintos Infinitos — Python no VS Code

Cada script é uma versão independente, convertida diretamente do notebook correspondente. As células foram mantidas com `# %%`. Os resultados antigos não são incorporados como saídas fabricadas.

## Windows

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Instale as extensões Python e Jupyter no VS Code para executar por células. Execute na ordem. Cada versão mantém o treinamento original; para reutilizar pesos compatíveis, execute as células anteriores à de treinamento, pule o treinamento e execute a célula de inferência com o checkpoint na pasta de trabalho.
