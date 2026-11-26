import random, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .model import MazeMiniGPT
from .dataset import carregar
from .tokenizer import VOCAB_SIZE

def treinar(steps=2000, lr=0.001, batch_size=32, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dados=carregar()
    def get_batch(split, batch_size):
        src=dados[split]; batch=src[torch.randint(len(src),(batch_size,))].to(device)
        return batch[:,:-1],batch[:,1:]
    get_batch('train',8) # preserva a chamada inicial da v0.3
    model=MazeMiniGPT().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    best=float('inf'); historico=[]
    Path('models').mkdir(exist_ok=True); Path('reports').mkdir(exist_ok=True)
    for step in range(1,steps+1):
        X,Y=get_batch('train',batch_size)
        loss=F.cross_entropy(model(X).reshape(-1,VOCAB_SIZE),Y.reshape(-1))
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step%250==0 or step==steps:
            model.eval()
            with torch.inference_mode():
                losses=[]
                for _ in range(25):
                    X,Y=get_batch('val',batch_size)
                    losses.append(F.cross_entropy(model(X).reshape(-1,VOCAB_SIZE),Y.reshape(-1)).item())
            model.train(); val=float(np.mean(losses))
            linha=dict(step=step,train_loss=loss.item(),val_loss=val); historico.append(linha); print(linha)
            if val<best:
                best=val; torch.save(dict(model_state=model.state_dict(),best_val=best),'models/best.pt')
    Path('reports/treinamento.json').write_text(json.dumps(historico,indent=2))

def carregar_modelo(path):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=MazeMiniGPT().to(device)
    model.load_state_dict(torch.load(path,map_location=device,weights_only=True)['model_state']); model.eval()
    return model
