from ultralytics import YOLO
import os
import cv2
import argparse

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

    def predict_video(self, video_path, conf=0.25, save_dir=None):
        if save_dir is None:
            save_dir = os.path.join(RUNS_DIR, "inference")
            
        results = self.model.predict(
            source=video_path,
            conf=conf,
            save=True,
            project=save_dir,
        )
        return results
    
    def predict_video_filtered(self, video_path, class_thresholds, default_conf=0.15, save_dir=None):
        if save_dir is None:
            save_dir = os.path.join(RUNS_DIR, "inference")
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        output_path = os.path.join(save_dir, "filtered_" + os.path.basename(video_path))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        results = self.model.predict(source=video_path, conf=default_conf, save=False, stream=True)

        for result in results:
            boxes = result.boxes
            keep_indices = []

            for j in range(len(boxes)):
                class_id = int(boxes.cls[j])
                class_name = self.model.names[class_id]
                confidence = float(boxes.conf[j])
                threshold = class_thresholds.get(class_name, default_conf)
                if confidence >= threshold:
                    keep_indices.append(j)

            result.boxes = boxes[keep_indices]

            annotated_frame = result.plot()
            writer.write(annotated_frame)

        writer.release()
        return output_path
    
def main():
    
    ## Path
    
    default_model = os.path.join(BASE_DIR, "..", "model", "best.pt")
    
    default_output = os.path.join(BASE_DIR, "..", "runs", "inference")
    
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--source", required=True, help="...")
    parser.add_argument("--mode", choices=["image","image_filtered", "video", "video_filtered"], default="image_filtered")    
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--output", default=default_output)
    args = parser.parse_args()
    
    detector = EPIDetector(args.model)

    class_thresholds = {
        "NO-Safety Helmet": 0.35,
        "NO-Safety Vest": 0.15,
        "Safety Helmet": 0.35,
        "Safety Vest": 0.3
    }
    
    if args.mode == "image":
        detector.predict_image(args.source, save_dir=args.output)
    elif args.mode == "image_filtered":
        detector.predict_image_filtered(args.source, class_thresholds, save_dir=args.output)
    elif args.mode == "video":
        detector.predict_video(args.source, save_dir=args.output)
    elif args.mode == "video_filtered":
        detector.predict_video_filtered(args.source, class_thresholds, save_dir=args.output)

if __name__ == "__main__":
    main()