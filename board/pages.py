from flask import Blueprint, render_template, request, redirect, url_for, jsonify
import os
from hazard.service import detect_hazard
import asyncio
from chatbot_merged.voice_commands import listen_for_commands, run_hazard_detection

# Create blueprint for all the "pages" routes
bp = Blueprint("pages", __name__, template_folder="templates")

# Folder for saving uploaded files (images/audio)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)     # ensure folder exists

# -------------------------
# ROUTES
# -------------------------

# Homepage
@bp.route("/")
def home():
    # Renders the homepage template
    return render_template("index.html")

# About page
@bp.route("/about")
def about():
    # Simple static route (can be extended later)
    return "Hello, About!"

# Hazard detection page
@bp.route("/detect", methods=["GET", "POST"])
def detect():
    """
    Handles hazard detection:
    - GET: Show upload form
    - POST: Accept an uploaded image, run the hazard detection model, 
      and return the annotated results.
    """
    result = None
    annotated_path = None

    if request.method =="POST":
        file = request.files["image"]
        if file:
            # Save uploaded file
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Path for the annotated output image
            annotated_path = os.path.join(UPLOAD_FOLDER, f"annotated_{file.filename}")
            
            # Run the detection model
            result = detect_hazard(filepath, save_path=annotated_path)

    return render_template("detect.html", result=result, annotated=annotated_path)
    

# Voice commands page
@bp.route("/voice", methods=["GET", "POST"])
def voice():
    """
    Handles voice input:
    - GET: Show voice recording interface
    - POST: Accept uploaded audio, save it, run transcription + pipeline,
      and return the transcript
    """
    transcript = None
    if request.method == "POST":
        audio = request.files["audio"]

        if audio:
            # Save the uploaded audio as captured.wav
            filepath = "uploads/captured.wav"
            audio.save(filepath)

            # Run transpraction + command pipeline asynchronously
            transcript = asyncio.run(listen_for_commands(filepath))

    return render_template("voice.html", transcript=transcript)

# Trigger hazard detection manually
@bp.route("/run_hazard")
def run_hazard():
    """
    Shortcut route to run hazard detection logic directly,
    then redirect back to the detect page
    """
    run_hazard_detection()
    return redirect(url_for("pages.detect"))