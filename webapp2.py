from flask import Flask, render_template, request, Response, redirect, url_for
import os
import cv2
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import time

app = Flask(__name__)

# Set upload and model directories
UPLOAD_FOLDER = os.getcwd()
MODEL_FOLDER = os.path.join(os.getcwd(), 'models')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Route for home page with model selection
@app.route("/", methods=["GET", "POST"])
def index():
    models = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pt')]  # List all YOLO models
    selected_model = None
    if request.method == "POST":
        # Handle form submission
        selected_model = request.form.get("model")
        if 'file' in request.files:
            f = request.files['file']
            filename = secure_filename(f.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(file_path)

            # Check file type
            file_extension = filename.rsplit('.', 1)[1].lower()

            if file_extension == 'jpg':  # Image file
                result_path = process_image(file_path, selected_model)
                return render_template("index.html", models=models, result_path=result_path, selected_model=selected_model)

            elif file_extension == 'mp4':  # Video file
                result_path = process_video(file_path, selected_model)
                return render_template("index.html", models=models, result_path=result_path, selected_model=selected_model)

    return render_template("index.html", models=models, selected_model=selected_model)


# Helper function to process an image
def process_image(image_path, model_name):
    model_path = os.path.join(MODEL_FOLDER, model_name)
    model = YOLO(model_path)
    results = model(image_path, save=True)

    # Retrieve the latest output folder from YOLO
    # output_folder = results[0].path.parent
    output_folder = '/'
    output_image = os.path.join(output_folder, os.listdir(output_folder)[0])
    return output_image


# Helper function to process a video
def process_video(video_path, model_name):
    model_path = os.path.join(MODEL_FOLDER, model_name)
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error opening video file"

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(UPLOAD_FOLDER, f"output_{int(time.time())}.mp4")
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

    
    return output_path


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)