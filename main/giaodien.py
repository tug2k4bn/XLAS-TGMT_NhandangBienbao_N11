import gradio as gr
import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import json
import random

# ==================== CẤU HÌNH (GIỮ NGUYÊN) ====================
MODEL_PATH = r'C:\Users\Admin\Downloads\final_model.pth'
CLASS_MAPPING_PATH = r'D:\xla v1\model_output\class_mapping\class_mapping_V1.json'
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
EXAMPLE_IMAGE_DIR = "examples"
# ==============================================================================


# --- KHỞI TẠO MÔ HÌNH (GIỮ NGUYÊN) ---
print("--- Đang khởi tạo ứng dụng ---")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Sử dụng thiết bị: {str(device).upper()}")
try:
    with open(CLASS_MAPPING_PATH, 'r', encoding='utf-8') as f:
        class_info = json.load(f)
        idx_to_class = {int(k): v for k, v in class_info['idx_to_class'].items()}
        num_classes = class_info['num_classes']
    print(f"Đã tải thành công {num_classes} lớp.")
except Exception as e:
    print(f"LỖI: Không thể tải file class mapping: {e}")
    exit()
try:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes + 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Đã tải mô hình thành công.")
except Exception as e:
    print(f"LỖI: Không thể tải mô hình: {e}")
    exit()
transform = transforms.Compose([transforms.ToTensor()])
COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(num_classes + 1)]
print("--- Khởi tạo hoàn tất. Giao diện sẵn sàng! ---")


def process_and_display(input_image, confidence_threshold):
    if input_image is None:
        return None, "Vui lòng chọn một ảnh.", ""

    pil_image = Image.fromarray(input_image)
    image_tensor = transform(pil_image).to(device)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        prediction = model(image_tensor)
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    output_img_np = input_image.copy()
    detections_count = 0
    detected_objects_list = []

    for i in range(len(boxes)):
        score = scores[i]
        if score > confidence_threshold:
            detections_count += 1
            box = boxes[i].astype(int)
            label_idx = labels[i]
            class_name = idx_to_class.get(label_idx, f'Unknown_{label_idx}')
            accuracy_percent = int(score * 100)
            detected_objects_list.append(f"- **{class_name}**: {accuracy_percent}%")
            label_text = f"{class_name}: {score:.2f}"
            color = COLORS[label_idx % len(COLORS)]
            cv2.rectangle(output_img_np, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(output_img_np, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    status_message = f"✅ Hoàn tất! Đã phát hiện {detections_count} đối tượng với ngưỡng tin cậy > {confidence_threshold:.2f}"

    if detected_objects_list:
        detection_summary = "\n".join(detected_objects_list)
    else:
        detection_summary = "Không có đối tượng nào được phát hiện."

    return output_img_np, status_message, detection_summary


def show_preview(image):
    if image is None:
        return None, "Vui lòng chọn ảnh để bắt đầu.", ""
    return image, "Ảnh đã sẵn sàng. Nhấn nút để nhận diện.", ""


# --- CSS (GIỮ NGUYÊN) ---
css = """
#gradio-app { background: linear-gradient(45deg, #0f0c29, #302b63, #24243e); }
.title_container h1 { background: -webkit-linear-gradient(45deg, #ff00ff, #00ffff, #ffdd00); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3.5em !important; text-align: center; font-weight: 900; text-shadow: 0 0 15px rgba(255, 255, 255, 0.2); }
.title_container p { text-align: center; color: #c0c0c0; font-size: 1.1em; }
#main_block { background: rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.18); border-radius: 20px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); padding: 20px !important; }
.gradio-container { color: #ffffff !important; }
.gr-label { color: #e0e0e0 !important; font-weight: bold !important; }
.gr-input, .gr-output { border-radius: 10px !important; }
.gr-button { border-radius: 8px !important; transition: all 0.2s ease-in-out !important; }
#predict_button { background: linear-gradient(45deg, #da22ff, #9733ee) !important; color: white !important; font-weight: bold !important; border: none !important; }
#predict_button:hover { transform: scale(1.05); box-shadow: 0 0 20px #da22ff; }
input[type="range"] { accent-color: #9733ee !important; }
#image_output img { border-radius: 15px; border: 2px solid #9733ee; box-shadow: 0 0 15px #9733ee; }
#image_output { background: transparent !important; border: none !important; }
.footer { text-align: center; margin-top: 20px; color: #a0a0a0; font-size: 0.9em; }
#detection_summary { background-color: rgba(0,0,0,0.2); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); }
"""

# --- XÂY DỰNG GIAO DIỆN GRADIO ---
with gr.Blocks(theme=gr.themes.Monochrome(), css=css, title="AI Object Detection") as app:
    with gr.Column(elem_classes="title_container"):
        gr.Markdown(
            """
            <h1>Nhận diện biển báo giao thông_Nhóm 11</h1>

            """
        )

    with gr.Blocks(elem_id="main_block"):
        with gr.Row(variant="panel"):
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("## 📤 Tải ảnh & Cài đặt")
                image_input = gr.Image(type="numpy", label="Chọn ảnh để nhận diện")
                with gr.Accordion("🛠️ Tùy chọn nâng cao", open=False):
                    confidence_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=DEFAULT_CONFIDENCE_THRESHOLD, step=0.05,
                        label="Ngưỡng tin cậy (Confidence Threshold)"
                    )
                status_output = gr.Textbox(label="📊 Trạng thái", interactive=False, lines=2,
                                           value="Vui lòng chọn ảnh để bắt đầu.")
                predict_button = gr.Button("Bắt đầu Nhận diện", variant="primary", elem_id="predict_button")

            with gr.Column(scale=2, min_width=500):
                gr.Markdown("## 🖼️ Kết quả Nhận diện")
                image_output = gr.Image(label="Kết quả (Xem trước)", interactive=False, elem_id="image_output")
                detection_summary_output = gr.Markdown(label="📝 Danh sách Đối tượng", elem_id="detection_summary")

        example_list = [os.path.join(EXAMPLE_IMAGE_DIR, f) for f in os.listdir(EXAMPLE_IMAGE_DIR)] if os.path.exists(
            EXAMPLE_IMAGE_DIR) else []
        if example_list:
            gr.Examples(
                examples=example_list,
                inputs=image_input,
                label="💡 Bấm để thử với các ảnh ví dụ"
            )

    with gr.Column(elem_classes="footer"):
        gr.Markdown("<p>Được phát triển bởi các chiến thần xử lí ảnh mạnh nhất thế giới</p>")

    image_input.change(
        fn=show_preview,
        inputs=image_input,
        outputs=[image_output, status_output, detection_summary_output]
    )

    predict_button.click(
        fn=process_and_display,
        inputs=[image_input, confidence_slider],
        outputs=[image_output, status_output, detection_summary_output],
        api_name="predict"
    )

app.launch(share=True, debug=True)