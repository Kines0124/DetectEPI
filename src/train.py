from ultralytics import YOLO

def train_model(model_size ="n", epochs = 100):
    model = YOLO(f"yolov8{model_size}.pt")
    
    results = model.train(
        data="data/data.yaml",
        epochs=epochs, ## Numero de  vezes que o modelo ve o dataset completo // Nos testes um valor entre 60-70 é o ideal, o valor 100 tem um leve overfitting
        imgsz=640, ## Tamanhodas imagens
        batch=-1,  ## Quantidade de imagens por vez: -1 para autobatch -> Decide a quatidade sozinho
        workers=8, 
        device=0,  ## Definindo pra usar GPU
        patience=20,
        project="C:/Projetos/DetectEPI/runs",
        name=f"detectepi_{model_size}_epochs{epochs}"
    )
    
def main ():
    train_model(model_size="s", epochs=100)
    ## train_model(model_size="n", epochs=100)

if __name__ == "__main__":
    main()
    
    