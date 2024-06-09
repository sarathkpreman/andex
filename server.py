import os
from flask import Flask, request, jsonify
from pathlib import Path
import easyocr
import csv
import cv2
import torch
import pandas as pd
from datetime import datetime
from omegaconf import DictConfig
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from Levenshtein import distance
import re

app = Flask(__name__)

# Define output directory
OUTPUT_DIR = "D:/andex/output/"

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Define DEFAULT_CONFIG object
DEFAULT_CONFIG = DictConfig({
    "model": "yolov8n.pt",  # Default YOLO model path
    "imgsz": 640,  # Default input image size
    "project": OUTPUT_DIR,
    "name": "run1",
    "exist_ok": True,  # Whether it's okay if the directory already exists
    "save": False,  # Whether saving is enabled or not
    "conf": 0.3,  # Confidence threshold
    "data": {},  # Placeholder for data-related configurations
    "device": "cpu",  # Default device to run the model
    "half": False,  # Whether to use half precision inference
    "dnn": None,  # Deep neural network backend
    "vid_stride": 1,  # Video stride
    "visualize": True,
    "augment": True,  # or false, depending on your preference
    "iou": 0.7,  # or any other desired value for Intersection over Union threshold
    "agnostic_nms": True,  # or false, depending on your desired behavior
    "max_det": 100,  # Default maximum number of detections
    "line_thickness": 3,  # Default line thickness for annotations
    "show": True,  # Whether to display results or not
    "save_crop": False,  # Whether to save crops or not
    "save_txt": False
    # Other configuration parameters...
})

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

previous_frames = {}  # Define previous_frames dictionary here

# Function to extract details from the number plate
def getOCR(im):
    gray_plate = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rgb_plate = cv2.cvtColor(gray_plate, cv2.COLOR_GRAY2RGB)
    results = reader.readtext(rgb_plate, detail=0, paragraph=True)
    text = ""
    state_code_found = False
    for result in results:
        # Check if the text matches the number plate format
        if re.match(r'^[ANAPARASBRCGCHDDDLGAGJHPHRJHJKKAKLLAMHMLMNMPMZNLODPBPYRJSKTGTNTRUKUPWB]{2}\d{2}[A-HJ-NP-Za-hj-np-z]{1,3}\d{4}$', result.strip()):
            # Ensure all letters are uppercase and add proper gaps
            formatted_text = result.strip().upper()
            formatted_text = formatted_text[:2] + ' ' + formatted_text[2:4] + ' ' + formatted_text[4:7] + ' ' + formatted_text[7:]
            # Insert a space between the third and fourth characters of the last part
            formatted_text = formatted_text[:9] + ' ' + formatted_text[9:]
            text = formatted_text
            break
    return text


# Track the positions of extracted data across consecutive frames
class PositionTracker:
    def __init__(self):
        self.prev_positions = None

    def track_positions(self, current_positions):
        if self.prev_positions is None:
            self.prev_positions = current_positions
            return True
        
        # Compare current positions with previous positions
        consistent = current_positions == self.prev_positions
        self.prev_positions = current_positions
        return consistent

# Initialize position tracker
position_tracker = PositionTracker()

class DetectionPredictor(BasePredictor):
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        self.txt_path = str(Path(self.save_dir) / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        
        # Initialize CSV writer outside the loop
        csv_file_path = f"{OUTPUT_DIR}/{self.data_path.stem}_number_plate_details.csv"

        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Get current date and time
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_time = datetime.now().strftime("%H:%M:%S")
            # Iterate over each detection and extract number plate details
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{self.model.names[c]} {conf:.2f}'
                # Get ROI from the frame
                roi = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                # Extract details from the number plate ROI
                plate_details = getOCR(roi)
                if plate_details:
                    # Check if the current vehicle is significantly different from previously detected ones across frames
                    skip = False
                    for prev_label, prev_details in previous_frames.items():
                        if distance(plate_details, prev_details) <= 2.1:
                            skip = True
                            break
                    if not skip:
                        # Update dictionary with current vehicle details
                        previous_frames[label] = plate_details
                        # Write plate details along with the current date and time as a row in the CSV file
                        csv_writer.writerow([plate_details, current_date, current_time])
                        log_string += f"{plate_details} OCR saved to {csv_file_path}, "
                    else:
                        log_string += f"Skipped saving {plate_details} OCR as redundant, "
                else:
                    log_string += f"No text detected from number plate {label}, "
                
                # Add bounding box to image
                if self.args.save or self.args.save_crop or self.args.show:
                    self.annotator.box_label(xyxy, label, color=colors(c, True))
                if self.args.save_crop:
                    imc = im0.copy()
                    save_one_box(xyxy,
                                imc,
                                file=Path(self.save_dir) / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}_{c}.jpg',
                                BGR=True)

        return log_string

def process_video(video_path, save_dir, frame_stride):
    cfg = DictConfig(DEFAULT_CONFIG)
    cfg.model = str(cfg.model) if isinstance(cfg.model, Path) else cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = str(video_path)  # Convert video path to string
    cfg.save_dir = save_dir  # Use provided save directory path
    cfg.show = False
    cfg.vid_stride = frame_stride  # Set frame stride

    predictor = DetectionPredictor(cfg)
    predictor()

def detect_from_live_camera(frame_stride):
    cfg = DictConfig(DEFAULT_CONFIG)
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = 0  # Use default camera
    cfg.vid_stride = frame_stride  # Set frame stride

    predictor = DetectionPredictor(cfg)
    predictor()

@app.route('/process_video', methods=['POST'])
def process_video_route():
    # Receive video file and save_dir from client
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    video_file = request.files['file']
    save_dir = request.form.get('save_dir', OUTPUT_DIR)  # Default save_dir is OUTPUT_DIR
    frame_stride = int(request.form.get('frame_stride', 28))  # Frame stride, default is 1
    
    video_path = 'video_output\\test1.mp4'  # Save the video temporarily
    video_file.save(video_path)

    # Process the video with frame sampling
    process_video(video_path, save_dir, frame_stride)
    
    # Construct the CSV file path
    csv_file_path = f"{save_dir}/{Path(video_path).stem}_number_plate_details.csv"

    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(csv_file_path, names=['Number Plate Details', 'Date', 'Time'])
    except FileNotFoundError:
        return jsonify({'error': 'CSV file not found'}), 404
    
    # Convert DataFrame to list of dictionaries
    data = df.to_dict(orient='records')

    # Delete the temporary video file
    Path(video_path).unlink()

    return jsonify(data), 200, {'Content-Type': 'application/json'}

if __name__ == "__main__":
    app.run(debug=True, port=5010, host="0.0.0.0")
