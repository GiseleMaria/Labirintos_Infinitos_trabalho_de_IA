import argparse
from src.dataset import pipeline

def main():
    parser=argparse.ArgumentParser(description='Labirintos Infinitos v0.5')
    parser.add_argument('etapa',choices=['dados','treinar','avaliar','comparar','gerar'])
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--quantidade',type=int)
    parser.add_argument('--checkpoint',default='models/best.pt')
    parser.add_argument('--sem-parquet',action='store_true')
    parser.add_argument('--entrada',type=int,choices=range(10),default=4)
    parser.add_argument('--temperatura',type=float,default=1.0)
    parser.add_argument('--sem-janela',action='store_true')
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--patience',type=int,default=2)
    parser.add_argument('--experimentos',action='store_true')
    args=parser.parse_args()
    if not 0 < args.temperatura < float('inf') or not 0 < args.lr < float('inf') or args.patience < 1: parser.error('Parametros invalidos')
    if args.quantidade is not None and args.quantidade < 1: parser.error('quantidade deve ser positiva')
    if args.steps < 1: parser.error('steps deve ser positivo')
    if args.etapa=='dados': pipeline(args.quantidade or 1500,parquet=not args.sem_parquet)
    elif args.etapa=='treinar':
        from src.train import treinar
        treinar(args.steps,lr=args.lr,patience=args.patience,checkpoint=args.checkpoint) if not args.experimentos else __import__('src.train',fromlist=['experimentos']).experimentos(args.steps,args.patience)
    else:
        from src.train import carregar_modelo
        model=carregar_modelo(args.checkpoint)
        if args.etapa=='avaliar':
            from src.evaluate import avaliar
            avaliar(model,args.quantidade or 100)
        elif args.etapa=='comparar':
            from src.evaluate import comparar
            comparar(model,args.quantidade or 50)
        else:
            from src.inference import gerar_chunk_ia
            from src.visualization import desenhar_chunks
            import matplotlib.pyplot as plt
            grids=[]; entradas=[]; saidas=[]; entrada=args.entrada
            for _ in range(args.quantidade or 3):
                grid,saida,tentativas=gerar_chunk_ia(model,entrada,temperature=args.temperatura)
                grids.append(grid); entradas.append(entrada); saidas.append(saida)
                print(entrada,'->',saida,'tentativas:',tentativas); entrada=saida
            from pathlib import Path
            Path('reports').mkdir(exist_ok=True)
            fig=desenhar_chunks(grids,entradas,saidas)
            fig.savefig('reports/geracao.png')
            if not args.sem_janela: plt.show()
            plt.close(fig)

if __name__=='__main__': main()
