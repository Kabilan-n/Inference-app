from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import time

app = Flask(__name__)

# Set upload and model directories
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')  # Files served from 'static/uploads'
UPDATE_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')  # Files served from 'static/uploads'
OUTPUT_FOLDER = os.path.join(os.getcwd(),'uploads')  # Files served from 'static/uploads'
MODEL_FOLDER = os.path.join(os.getcwd(), 'models')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    # Retrieve available models
    models = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pt')]
    selected_model = None
    result_path = None
    result_type = None

    if request.method == "POST":
        selected_model = request.form.get("model")
        if 'file' in request.files:
            # Save uploaded file
            f = request.files['file']
            filename = secure_filename(f.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(file_path)

            # Process file based on type
            file_extension = filename.rsplit('.', 1)[1].lower()
            if file_extension == 'jpg':  # Image file
                result_path = process_image(file_path, selected_model)
                result_type = 'image'
            elif file_extension == 'mp4':  # Video file
                result_path = process_video(file_path, selected_model)
                result_type = 'video'

    return render_template(
        "index.html",
        models=models,
        selected_model=selected_model,
        result_path=result_path,
        result_type=result_type,
    )

# Helper function to process an image
def process_image(image_path, model_name):
    model_path = os.path.join(MODEL_FOLDER, model_name)
    model = YOLO(model_path)
    results = model(image_path, save=True)

    # Locate YOLO's saved output file
    output_folder = os.path.dirname(results[0].path)
    output_image = os.path.join(output_folder, os.listdir(output_folder)[0])

    # Move result to static/uploads for serving
    final_path = os.path.join(UPDATE_FOLDER, f"result_{int(time.time())}.jpg")
    os.rename(output_image, final_path)
    return final_path.replace(os.getcwd() + "/", "")  # Relative path for serving

# Helper function to process a video
def process_video(video_path, model_name):
    model_path = os.path.join(MODEL_FOLDER, model_name)
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error opening video file"

    # Video output setup
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = os.path.join(UPDATE_FOLDER, f"result_{int(time.time())}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame)
        res_plotted = results[0].plot()
        out.write(res_plotted)

    cap.release()
    out.release()
    return output_path.replace(os.getcwd() + "/", "../")  # Relative path for serving


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)