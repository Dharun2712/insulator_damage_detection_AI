from flask import Flask, render_template, request, send_from_directory, url_for
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

print("Starting the application...")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

print("Loading CLIP model... This may take a few minutes on first run as it downloads the model (~605MB)")
try:
    # Load pretrained CLIP model (no training needed)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="./model_cache")
    print("Model downloaded successfully!")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="./model_cache")
    print("Processor downloaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    filename = None

    if request.method == 'POST':
        if 'file' not in request.files:
            result = "No file uploaded."
        else:
            file = request.files['file']
            if file.filename != '':
                # Save with a secure filename
                import secrets
                file_extension = os.path.splitext(file.filename)[1]
                secure_filename = secrets.token_hex(16) + file_extension
                filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename)
                file.save(filename)
                
                # Create URL for the uploaded file
                image_url = url_for('uploaded_file', filename=secure_filename)

                # Load image and predict
                image = Image.open(filename).convert("RGB")
                texts = [
                    "a photo of a healthy outdoor insulator",
                    "a photo of a damaged or broken outdoor insulator"
                ]

                inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
                outputs = model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
                probs = probs.detach().numpy()[0]

                healthy_prob, damaged_prob = probs[0], probs[1]

                if healthy_prob > damaged_prob:
                    result = "✅ Insulator is NORMAL — no repair required."
                else:
                    result = "⚠️ Insulator is DAMAGED — maintenance/repair required."

    return render_template('index.html', result=result, filename=image_url if 'image_url' in locals() else None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    print("Starting Flask server...")
    try:
        # Explicitly set host to 0.0.0.0 to allow external connections
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        input("Press Enter to exit...")