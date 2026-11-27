import random, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .model import MazeMiniGPT
from .dataset import carregar
from .tokenizer import VOCAB_SIZE

def treinar(steps=2000, lr=0.001, batch_size=32, seed=42, patience=2, checkpoint="models/best.pt"):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dados=carregar()
    def get_batch(split, batch_size):
        src=dados[split]; batch=src[torch.randint(len(src),(batch_size,))].to(device)
        return batch[:,:-1],batch[:,1:]
    get_batch('train',8) # preserva a chamada inicial da v0.3
    model=MazeMiniGPT().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    best=float('inf'); historico=[]; espera=0
    import mlflow
    import os
    if not os.environ.get('MLFLOW_TRACKING_URI'):
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('Labirintos Infinitos')
    with mlflow.start_run():
        mlflow.log_params(dict(lr=lr,steps=steps,batch_size=batch_size,seed=seed,patience=patience,device=str(device)))
        return _loop(model,optimizer,dados,get_batch,steps,batch_size,patience,checkpoint,device,mlflow)

def _loop(model,optimizer,dados,get_batch,steps,batch_size,patience,checkpoint,device,mlflow):
    best=float('inf'); historico=[]; espera=0
    Path('models').mkdir(exist_ok=True); Path('reports').mkdir(exist_ok=True)
    for step in range(1,steps+1):
        X,Y=get_batch('train',batch_size)
        loss=F.cross_entropy(model(X).reshape(-1,VOCAB_SIZE),Y.reshape(-1))
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step%250==0 or step==steps:
            model.eval()
            with torch.inference_mode():
                losses=[]
                for batch in dados['val'].split(batch_size):
                    X,Y=batch[:,:-1].to(device),batch[:,1:].to(device)
                    losses.append(F.cross_entropy(model(X).reshape(-1,VOCAB_SIZE),Y.reshape(-1),reduction='sum').item())
            model.train(); val=float(sum(losses)/(len(dados['val'])*100))
            linha=dict(step=step,train_loss=loss.item(),val_loss=val); historico.append(linha); print(linha)
            mlflow.log_metrics(dict(train_loss=loss.item(),val_loss=val),step=step)
            if val<best:
                best=val; espera=0
                Path(checkpoint).parent.mkdir(parents=True,exist_ok=True)
                torch.save(dict(model_state=model.state_dict(),best_val=best,step=step),checkpoint)
            else:
                espera+=1
                if espera>=patience:
                    print("Early stopping",step); break
    nome=Path(checkpoint).stem
    rel=Path(f'reports/treinamento_{nome}.json')
    rel.write_text(json.dumps(historico,indent=2))
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    for k in ('train_loss','val_loss'):
        ax.plot([i['step'] for i in historico],[i[k] for i in historico],label=k)
    ax.legend(); ax.set_xlabel('Step'); ax.set_ylabel('Loss')
    imagem=rel.with_suffix('.png'); fig.savefig(imagem); plt.close(fig)
    mlflow.log_metric('best_val',best)
    for arquivo in (rel,imagem,Path(checkpoint)): mlflow.log_artifact(str(arquivo))
    return best

def experimentos(steps=2000,patience=2):
    import shutil
    resultados=[]
    for lr in (0.001,0.0003,0.0001):
        path=f'models/experimento_{lr}.pt'
        val=treinar(steps,lr=lr,patience=patience,checkpoint=path)
        resultados.append(dict(lr=lr,best_val=val,checkpoint=path))
    melhor=min(resultados,key=lambda r:r['best_val'])
    shutil.copy2(melhor['checkpoint'],'models/best.pt')
    Path('reports/experimentos.json').write_text(json.dumps(dict(runs=resultados,melhor=melhor),indent=2))
    print(melhor)

def carregar_modelo(path):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=MazeMiniGPT().to(device)
    model.load_state_dict(torch.load(path,map_location=device,weights_only=True)['model_state']); model.eval()
    return model
