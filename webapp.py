from flask import Flask, render_template, request, send_file, Response, send_from_directory
import os
import cv2
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import time

app = Flask(__name__)

# Set upload and model directories
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')  # Files served from 'static/uploads'
DOWNLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'downloads') 
MODEL_FOLDER = os.path.join(os.getcwd(), 'models')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    models = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pt')]
    selected_model = None
    result_path = None
    result_type = None

    if request.method == "POST":
        selected_model = request.form.get("model")
        if 'file' in request.files:
            f = request.files['file']
            filename = secure_filename(f.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(file_path)

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

@app.route("/video/<path:filename>")
def stream_video(filename):
    def generate():
        with open(os.path.join(UPLOAD_FOLDER, filename), "rb") as f:
            while chunk := f.read(8192):
                yield chunk

    headers = {"Content-Type": "video/mp4"}
    return Response(generate(), headers=headers)

# Helper function to process an image
# def process_image(image_path, model_name):
#     model_path = os.path.join(MODEL_FOLDER, model_name)
#     model = YOLO(model_path)
#     results = model(image_path, save=True)

#     output_folder = os.path.dirname(results[0].path)
#     output_image = os.path.join(output_folder, os.listdir(output_folder)[0])
#     final_path = os.path.join(UPLOAD_FOLDER, f"result_{int(time.time())}.jpg")
#     os.rename(output_image, final_path)
#     return os.path.basename(final_path)

def process_image(image_path, model_name):
    model_path = os.path.join(MODEL_FOLDER, model_name)
    model = YOLO(model_path)
    results = model(image_path, save=True)

    annotated_image = results[0].plot()

    # Locate YOLO's saved output file
    output_folder = os.path.dirname(results[0].path)
    output_file = os.listdir(output_folder)[0]  # Get the output file name
    output_image = os.path.join(output_folder, output_file)


    # Extract the original file extension
    file_extension = os.path.splitext(output_file)[1]
    
    # Create a unique name for the output image
    output_filename = f"result_{int(time.time())}{file_extension}"
    final_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
    print("look here!")
    print(output_image,final_path)
    cv2.imwrite(final_path, annotated_image)
    # Move the file to the desired location
    # os.rename(output_image, final_path)
    
    # Return the relative path of the result
    return os.path.basename(final_path)

@app.route('/download/<path:filename>')
def download_file_image(filename):
    directory = os.path.join(DOWNLOAD_FOLDER)
    try:
        return send_from_directory(directory, filename, as_attachment=True)
    except Exception as e:
        # logging.error(f"Error downloading file: {e}")
        return "File not found", 404

# def process_image(image_path, model_name):
#     model_path = os.path.join(MODEL_FOLDER, model_name)
#     model = YOLO(model_path)
#     results = model(image_path, save=True)

#     # Locate YOLO's saved output file
#     output_folder = os.path.dirname(results[0].path)
#     output_image = os.path.join(output_folder, os.listdir(output_folder)[0])

#     # Move result to static/uploads for serving
#     output_filename = f"result_{int(time.time())}.jpg"
#     final_path = os.path.join(UPLOAD_FOLDER, output_filename)
#     os.rename(output_image, final_path)
#     return os.path.relpath(final_path, UPLOAD_FOLDER)  # Relative path for serving


# Helper function to process a video
def process_video(video_path, model_name):
    model_path = os.path.join(MODEL_FOLDER, model_name)
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error opening video file"

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = os.path.join(DOWNLOAD_FOLDER, f"result_{int(time.time())}.mp4")
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
    return os.path.basename(output_path)

@app.route('/download/<path:filename>')
def download_file_video(filename):
    directory = os.path.join(DOWNLOAD_FOLDER)
    try:
        return send_from_directory(directory, filename, as_attachment=True)
    except Exception as e:
        # logging.error(f"Error downloading file: {e}")
        return "File not found", 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
