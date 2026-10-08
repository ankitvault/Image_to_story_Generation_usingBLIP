import os
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory

# Model pipeline must be imported
from models import pipeline

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # Maximum 32MB upload size

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/api/process", methods=["POST"])
def process():
    if not UPLOAD_FOLDER.exists():
        UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        
    if "images" not in request.files:
        return jsonify({"error": "No images provided"}), 400
        
    files = request.files.getlist("images")
    if not files or files[0].filename == "":
        return jsonify({"error": "No valid files provided"}), 400
        
    image_paths = []
    image_urls = []
    
    for file in files:
        if file and allowed_file(file.filename):
            ext = os.path.splitext(file.filename)[1].lower()
            filename = f"{uuid.uuid4().hex}{ext}"
            file_path = UPLOAD_FOLDER / filename
            file.save(file_path)
            
            image_paths.append(str(file_path))
            image_urls.append(f"/uploads/{filename}")
            
    if not image_paths:
        return jsonify({"error": "No valid image files uploaded"}), 400
        
    # Generate captions for each image
    captions = []
    for path in image_paths:
        cap = pipeline.generate_caption(path)
        captions.append(cap)
        
    # Generate story from the combined captions
    story = pipeline.generate_story(captions)
    
    # Generate short summary of the story
    summary = pipeline.summarize_story(story)
    
    return jsonify({
        "captions": captions,
        "story": story,
        "summary": summary,
        "image_urls": image_urls
    })

if __name__ == "__main__":
    if not UPLOAD_FOLDER.exists():
        UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        
    print("Server ready -> http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
