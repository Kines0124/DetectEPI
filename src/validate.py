from ultralytics import YOLO

def main():
    model = YOLO("C:/Projetos/DetectEPI/runs/detectepi_v2_classweights/weights/best.pt")

    metrics = model.val(
        data="C:/Projetos/DetectEPI/data/data.yaml",
        split="test",
        project="C:/Projetos/DetectEPI/runs",
        name="detectepi_v2_test_classweights_eval"
    )

if __name__ == "__main__":
    main()