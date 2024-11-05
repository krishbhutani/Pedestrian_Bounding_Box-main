from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from ultralytics import YOLO
import os
import cv2
from pathlib import Path
import sys

app = Flask(__name__)

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

model_path = PACKAGE_ROOT / "trained_model" / "best.pt"

# Load the trained YOLOv8 model (ensure this is your .pt file)
model = YOLO(model_path)

# Set up a folder to store uploaded images
UPLOAD_FOLDER = PACKAGE_ROOT / 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Homepage
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Perform YOLOv8 prediction
        results = model.predict(filepath)

        # Extract detected objects (for counting people)
        people_count = 0
        for result in results:
            # Assuming 'person' is the class name for people
            people_count += sum(1 for cls in result.boxes.cls if model.names[int(cls)] == 'person')

        # Load the original image using OpenCV
        image = cv2.imread(filepath)

        # Draw bounding boxes on the image
        for result in results[0].boxes:
            box = result.xyxy[0].cpu().numpy()  
            cls = int(result.cls[0])  # Get the class index
            label = model.names[cls]  # Get the class name
            if label == 'person':
                # Draw a rectangle for people
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                cv2.putText(image, label, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Save the resulting image
        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + file.filename)
        cv2.imwrite(result_image_path, image)

        # Redirect to display the result
        return render_template('result.html', people_count=people_count, image_file='result_' + file.filename)

# Serve uploaded files (images)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=8005)

