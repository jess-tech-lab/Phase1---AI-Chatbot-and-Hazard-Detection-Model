import cv2
from ultralytics import YOLO

# Load model once at import

MODEL_PATH = "runs/detect/train8/weights/best.pt"
model = YOLO(MODEL_PATH)     # Load trained YOLO model
labels = model.names         # get class labels

def detect_hazard(image_path, conf = 0.25, imgsz = 320, save_path = None):
    """
    Run YOLO hazard detection on a single image file and return results
    Args:
        image_path (str): Path to the input image file.
        conf (float): Confidence threshold for detections (default: 0.25).
        imgsz (int): Inference image size (default: 320).
        save_path (str): If provided, save an annotated image with bounding boxes.

    Returns:
        list[dict]: A list of hazards, where each hazard is represented as:
        {
            "label": str,               # Predicted class label
            "confidence": float         # Detection confidence score
            "bbox": [x1, y1, x2, y2]    # Bounding box coordinates
        }
    """

    # Read image using OpenCV
    frame = cv2.imread(image_path)
    if frame is None:
        return ValueError(f"Could not read image {image_path}")
    
    # Rune interference w/ YOLO (returns a list of results; take first item [0])
    results = model.predict(source=frame, imgsz=imgsz, conf=conf)[0]

    hazards = []

    # Iterate through detected boxes, classes, and confidence scores
    for box, cls, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        hazards.append({
            "label": labels[int(cls)]       # Convert class ID to label
            
        })