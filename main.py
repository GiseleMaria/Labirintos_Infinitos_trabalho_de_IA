import argparse
from src.dataset import pipeline

def main():
    parser=argparse.ArgumentParser(description='Labirintos Infinitos v0.4')
    parser.add_argument('etapa',choices=['dados','treinar','avaliar','comparar','gerar'])
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--quantidade',type=int)
    parser.add_argument('--checkpoint',default='models/best.pt')
    parser.add_argument('--sem-parquet',action='store_true')
    args=parser.parse_args()
    if args.quantidade is not None and args.quantidade < 1: parser.error('quantidade deve ser positiva')
    if args.steps < 1: parser.error('steps deve ser positivo')
    if args.etapa=='dados': pipeline(args.quantidade or 1500,parquet=not args.sem_parquet)
    elif args.etapa=='treinar':
        from src.train import treinar
        treinar(args.steps)
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
            grids=[]; entradas=[]; saidas=[]; entrada=4
            for _ in range(args.quantidade or 3):
                grid,saida,tentativas=gerar_chunk_ia(model,entrada)
                grids.append(grid); entradas.append(entrada); saidas.append(saida)
                print(entrada,'->',saida,'tentativas:',tentativas); entrada=saida
            desenhar_chunks(grids,entradas,saidas); plt.show()

if __name__=='__main__': main()
