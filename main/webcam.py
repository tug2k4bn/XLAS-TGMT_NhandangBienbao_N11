import gradio as gr
import os
import torch
import torchvision
from PIL import Image
import cv2
import numpy as np
import json

MODELS_CONFIG = {
    "Faster R-CNN": {
        "path": r'C:\Users\Admin\Downloads\final_model.pth',
        "class_mapping": r'D:\xla v1\model_output\class_mapping\class_mapping_V1.json'
    },
    "YOLOv5": {
        "path": r'C:\Users\Admin\Downloads\xla\yolov5\runs\train\exp7\weights\best.pt',
    }
}

YOLOV5_CLASS_NAMES_ENGLISH = [
    "regulation--keep-right", "regulation--height-limit", "warning--railroad-with-barrier",
    "warning--falling-rocks-right", "regulation--yield", "warning--right-turn",
    "regulation--pedestrians-only", "warning--pedestrian-crossing", "regulation--no-entry",
    "warning--slippery-road", "warning--left-turn", "information--parking",
    "information--bus-stop", "warning--intersection", "regulation--stop",
    "regulation--maximum-speed-limit", "regulation--turn-right", "warning--roundabout",
    "warning--speed-bump", "warning--bumpy-road", "warning--railroad-no-barrier",
    "regulation--bicycles-only", "regulation--yield-to-oncoming-traffic",
    "regulation--shared-pedestrian-bicycle-lane", "regulation--no-bicycles",
    "regulation--no-pedestrians", "regulation--no-overtaking", "regulation--keep-left",
    "regulation--go-straight", "regulation--no-parking", "regulation--no-right-turn",
    "regulation--no-left-turn", "supplementary--left-arrow", "regulation--no-heavy-trucks",
    "regulation--weight-limit", "regulation--no-u-turn", "warning--other-dangers",
    "warning--side-road-right", "warning--winding-road-right", "regulation--turn-left",
    "warning--road-work", "warning--children", "warning--merge-right",
    "warning--road-narrows-right", "information--highway", "regulation--pass-either-side",
    "warning--bicycle-crossing", "supplementary--safe-distance", "regulation--straight-or-right",
    "regulation--straight-or-left", "regulation--no-motor-vehicles", "supplementary--right-arrow",
    "information--disabled-person", "regulation--no-trucks", "regulation--roundabout-mandatory",
    "regulation--no-overweight-trucks", "warning--traffic-lights", "warning--road-narrows",
    "warning--road-narrows-left", "information--gas-station", "regulation--axle-weight-limit",
    "warning--road-marker-post", "warning--wild-animals", "warning--domestic-animals",
    "regulation--left-or-right-turn", "warning--steep-descent-left", "warning--winding-road-left",
    "regulation--no-buses", "warning--falling-rocks-left", "warning--road-marker-right",
    "warning--road-marker-left", "warning--side-road-left", "warning--uneven-road",
    "warning--steep-descent-right", "warning--merge-left", "information--hospital"
]

DEFAULT_CONFIDENCE_THRESHOLD = 0.5

print("--- Đang khởi tạo ứng dụng ---")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Sử dụng thiết bị: {str(device).upper()}")

loaded_models, class_mappings = {}, {}
COLORS = [
    (255, 87, 87), (87, 255, 87), (87, 87, 255), (255, 255, 87),
    (87, 255, 255), (255, 87, 255), (255, 128, 0), (0, 128, 255),
    (128, 0, 255), (255, 0, 128), (128, 255, 0), (0, 255, 128)
] * (len(YOLOV5_CLASS_NAMES_ENGLISH) // 12 + 1)
print("--- Khởi tạo hoàn tất. Giao diện sẵn sàng! ---")


def load_model_and_classes(model_name):
    if model_name in loaded_models: return loaded_models[model_name], class_mappings.get(model_name)
    print(f"Đang tải mô hình '{model_name}'...")
    config = MODELS_CONFIG[model_name]
    model, idx_to_class = None, {}
    try:
        if model_name == "Faster R-CNN":
            with open(config['class_mapping'], 'r', encoding='utf-8') as f:
                class_info = json.load(f)
                idx_to_class = {int(k): v for k, v in class_info['idx_to_class'].items()}
                num_classes = class_info['num_classes']
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes + 1)
            model.load_state_dict(torch.load(config['path'], map_location=device))
        elif model_name == "YOLOv5":
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=config['path'], force_reload=False)
            model.names = YOLOV5_CLASS_NAMES_ENGLISH
            idx_to_class = {i: name for i, name in enumerate(YOLOV5_CLASS_NAMES_ENGLISH)}
        if model:
            model.to(device)
            model.eval()
            loaded_models[model_name] = model
            class_mappings[model_name] = idx_to_class
            print(f"Đã tải thành công mô hình '{model_name}'.")
            return model, idx_to_class
    except Exception as e:
        print(f"LỖI: Không thể tải mô hình '{model_name}': {e}")
        return None, None

def predict_single_model(input_image, confidence_threshold, model_choice):
    model, idx_to_class = load_model_and_classes(model_choice)
    if model is None: return [], None
    raw_predictions = []
    if model_choice == "Faster R-CNN":
        pil_image = Image.fromarray(input_image)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        image_tensor = transform(pil_image).to(device).unsqueeze(0)
        with torch.no_grad():
            prediction = model(image_tensor)[0]
        for i in range(len(prediction['scores'])):
            score = prediction['scores'][i].item()
            if score > confidence_threshold:
                raw_predictions.append({'score': score, 'box': prediction['boxes'][i].cpu().numpy().astype(int), 'class_id': prediction['labels'][i].item()})
    elif model_choice == "YOLOv5":
        results = model(input_image)
        for pred in results.xyxy[0]:
            score = pred[4].item()
            if score > confidence_threshold:
                raw_predictions.append({'score': score, 'box': pred[:4].cpu().numpy().astype(int), 'class_id': int(pred[5].item())})
    return raw_predictions, idx_to_class

def detect_on_stream(frame, confidence_threshold, model_choice):
    if frame is None:
        return None

    predictions, idx_to_class = predict_single_model(frame, confidence_threshold, model_choice)

    if not idx_to_class:
        return frame

    output_frame_np = frame.copy()
    for pred in predictions:
        class_id = pred['class_id']
        class_name = idx_to_class.get(class_id, 'Unknown')
        score = pred['score']
        box = pred['box']

        color = COLORS[class_id % len(COLORS)]
        label_text = f"{class_name}: {score:.2f}"

        cv2.rectangle(output_frame_np, (box[0], box[1]), (box[2], box[3]), color, 2)

        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y_pos = box[1] - 10 if box[1] - 10 > 10 else box[1] + text_height + 10
        cv2.rectangle(output_frame_np, (box[0], text_y_pos - text_height - 5), (box[0] + text_width, text_y_pos + baseline - 5), color, -1)
        cv2.putText(output_frame_np, label_text, (box[0], text_y_pos - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return output_frame_np

css_webcam = """
#main_title { text-align: center; font-size: 2.5em; font-weight: bold; color: #3498db; }
.container { background: #f4f6f6; padding: 20px; border-radius: 15px; }
"""

with gr.Blocks(css=css_webcam, theme=gr.themes.Soft()) as webcam_app:
    gr.Markdown("<h1 id='main_title'>Nhận diện Biển báo Giao thông qua Webcam</h1>")

    with gr.Group():
        with gr.Row():
            model_choice_radio = gr.Radio(
                choices=["Faster R-CNN", "YOLOv5"],
                label="🤖 Chọn mô hình",
                value="YOLOv5"
            )
            confidence_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=DEFAULT_CONFIDENCE_THRESHOLD,
                step=0.05,
                label="Ngưỡng tin cậy"
            )

    with gr.Row(equal_height=True):
        with gr.Column():
            webcam_input = gr.Image(source="webcam", streaming=True, label="Đầu vào Webcam")
        with gr.Column():
            processed_output = gr.Image(label="Kết quả Nhận diện")

    webcam_input.stream(
        fn=detect_on_stream,
        inputs=[webcam_input, confidence_slider, model_choice_radio],
        outputs=processed_output
    )

webcam_app.launch(share=True, debug=True)
