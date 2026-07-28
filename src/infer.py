from ultralytics import YOLO
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, "..", "runs")

class EPIDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        

    def predict_image(self, image_path, conf=0.25, save_dir=None):
        if save_dir is None:
            save_dir = os.path.join(RUNS_DIR, "inference")
            
        results = self.model.predict(
            source=image_path,
            conf=conf,
            save=True,
            project=save_dir,
            name="predictions"
        )
        return results
    
    def predict_image_filtered(self, image_path, class_thresholds, default_conf=0.15, save_dir=None):
        if save_dir is None:
            save_dir = os.path.join(RUNS_DIR, "inference")

        results = self.model.predict(
            source=image_path,
            conf=default_conf,
            save=False
        )

        result = results[0]
        boxes = result.boxes

        keep_indices = []
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])
            class_name = self.model.names[class_id]
            confidence = float(boxes.conf[i])

            threshold = class_thresholds.get(class_name, default_conf)
            if confidence >= threshold:
                keep_indices.append(i)

        result.boxes = boxes[keep_indices]
        
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, "filtered_" + os.path.basename(image_path))
        result.save(filename=output_path)
        
        return result

    def predict_video(self, video_path, conf=0.25):
        
        return None


def main():
    model_path = os.path.join(BASE_DIR, "..", "runs", "detectepi_v1", "weights", "best.pt")
    detector = EPIDetector(model_path)
    
    image_path = os.path.join(BASE_DIR, "..", "data", "test", "images", "00411_jpg.rf.5834946a4dd017b28423c6150d115ecc.jpg")
    
    class_thresholds = {
        "NO-Safety Helmet": 0.4,
        "NO-Safety Vest": 0.15,
        "Safety Helmet": 0.4,
        "Safety Vest": 0.3
    }
    
    result = detector.predict_image_filtered(image_path, class_thresholds)


if __name__ == "__main__":
    main()