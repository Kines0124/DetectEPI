import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..","src"))

from infer import EPIDetector, filter_boxes_by_threshold

class FakeBoxes:
    def __init__(self, cls_list, conf_list):
        self.cls = cls_list
        self.conf = conf_list

    def __len__(self):
        return len(self.cls)

## Testando o threshold

def test_filter_boxes_by_thresholds():
    
    boxes = FakeBoxes(cls_list=[0, 0], conf_list=[0.5, 0.1])
    class_thresholds = {"Safety Helmet": 0.4}
    class_names = {0: "Safety Helmet"}
    
    resultado = filter_boxes_by_threshold(boxes, class_thresholds, class_names, default_conf=0.15)
    
    assert resultado == [0]

## Testando se o modelo carrega

def test_detector_loads_model():
    
    model_path = os.path.join(os.path.dirname(__file__), "..","model","best.pt")
    
    detector = EPIDetector(model_path)
    
    assert detector.model is not None
    
# Testando o pipeline

def test_pipeline():
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "best.pt")
    image_path = os.path.join(os.path.dirname(__file__), "..", "data", "inferTest", "Teste.png")
    output_path = os.path.join(os.path.dirname(__file__), "tmp_output")
    default_conf = 0.15
    
    detector = EPIDetector(model_path)
    
    class_thresholds = {
    "NO-Safety Helmet": 0.35,
    "NO-Safety Vest": 0.15,
    "Safety Helmet": 0.35,
    "Safety Vest": 0.3
    }
    
    result = detector.predict_image_filtered(image_path, class_thresholds, default_conf, output_path)
    
    assert result is not None