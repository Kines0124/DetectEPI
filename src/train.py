from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def main():
    results = model.train(
        data="data/data.yaml",
        epochs=100, ## Numero de  vezes que o modelo ve o dataset completo // Nos testes um valor entre 60-70 é o ideal, o valor 100 tem um leve overfitting
        imgsz=640, ## Tamanhodas imagens
        batch=32,  ## Quantidade de imagens por vez
        workers=8, 
        device=0,  ## Definindo pra usar GPU
        patience=20,
        project="C:/Projetos/DetectEPI/runs",
        name="detectepi_v2_classweights"
    )

if __name__ == "__main__":
    main()
    
    