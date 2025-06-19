from flask import Flask, request, render_template, send_file, send_from_directory, url_for
import os, shutil, uuid, cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.preprocessing import normalize
import hdbscan
from collections import defaultdict
from zipfile import ZipFile
import random

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
PREVIEW_FOLDER = 'static/previews'

if os.path.exists(UPLOAD_FOLDER):
    shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER)

if os.path.exists(RESULT_FOLDER):
    shutil.rmtree(RESULT_FOLDER)
    os.makedirs(RESULT_FOLDER)
    
if os.path.exists(PREVIEW_FOLDER):
    shutil.rmtree(PREVIEW_FOLDER)
    os.makedirs(PREVIEW_FOLDER)


face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

@app.route('/', methods=['GET', 'POST'])
def index():
    zip_url = None
    previews = []
    session_id = None

    if request.method == 'POST':
        # Clean previous previews
        shutil.rmtree(PREVIEW_FOLDER)
        os.makedirs(PREVIEW_FOLDER)

        session_id = str(uuid.uuid4())
        session_upload_dir = os.path.join(UPLOAD_FOLDER, session_id)
        os.makedirs(session_upload_dir)

        embeddings = []
        paths = []

        for file in request.files.getlist('photos'):
            file_path = os.path.join(session_upload_dir, file.filename)
            file.save(file_path)

            img = cv2.imread(file_path)
            if img is None:
                continue

            faces = face_app.get(img)
            for face in faces:
                if face.embedding is not None:
                    embeddings.append(face.embedding)
                    paths.append(file_path)

        if not embeddings:
            return render_template('index.html', zip_url=None, previews=[],message="Not valid faces detected.")
        if len(embeddings)<2:
            return render_template('index.html', message="Please upload at least two images with faces.")

#cluster embeddings
        embeddings = normalize(np.array(embeddings))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2,min_samples=1)
        labels = clusterer.fit_predict(embeddings)

        clusters = defaultdict(list)
        for label, path in zip(labels, paths):
            clusters[label].append(path)
        for label in clusters:
            clusters[label] = list(set(clusters[label]))

        result_dir = os.path.join(RESULT_FOLDER, session_id)
        os.makedirs(result_dir)

        for label, img_paths in clusters.items():
            person_dir = os.path.join(result_dir, f"person_{label}" if label != -1 else "unknown")
            os.makedirs(person_dir, exist_ok=True)
            for img_path in img_paths:
                shutil.copy(img_path, person_dir)
                
            # 1. Try to find a photo with only one face
            selected_preview = None
            for img_path in img_paths:
                img = cv2.imread(img_path)
                faces = face_app.get(img)
                if len(faces) == 1:
                    selected_preview = img_path
                    break

            # 2. If none found, just pick the last one (or random)
            if not selected_preview:
                selected_preview = random.choice(img_paths)  # fallback

            # 3. Save preview
            preview_name = f"person_{label}.jpg"
            preview_path = os.path.join(PREVIEW_FOLDER, preview_name)
            shutil.copy(selected_preview, preview_path)
            previews.append(f"previews/{preview_name}")

   
        # ZIP creation
        zip_path = os.path.join(RESULT_FOLDER, f"{session_id}.zip")
        with ZipFile(zip_path, 'w') as zipf:
            
            for root, dirs, files in os.walk(result_dir):
                for file in files:
                    filepath = os.path.join(root, file)
                    zipf.write(filepath, os.path.relpath(filepath, result_dir))

        zip_url = f"/download/{session_id}.zip"

    return render_template('index.html', previews=previews, session_id=session_id)


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)

@app.route('/static/previews/<filename>')
def send_preview(filename):
    return send_from_directory(PREVIEW_FOLDER, filename)

@app.route('/finalize_and_download', methods=['POST'])
def finalize_and_download():
    old_names = request.form.getlist('old_names')
    new_names = request.form.getlist('new_names')
    session_id = request.form.get('session_id')

    result_dir = os.path.join(RESULT_FOLDER, session_id)

    for old, new in zip(old_names, new_names):
        label = old.split('/')[-1].replace("person_", "").replace(".jpg", "")
        old_folder = os.path.join(result_dir, f"{'unknown' if label == 'unknown' else 'person_' + label}")
        new_folder = os.path.join(result_dir, new.replace(" ", "_"))
        if os.path.exists(old_folder):
            os.rename(old_folder, new_folder)

        old_preview = os.path.join(PREVIEW_FOLDER, f"person_{label}.jpg")
        new_preview = os.path.join(PREVIEW_FOLDER, f"{new.replace(' ', '_')}.jpg")
        if os.path.exists(old_preview):
            os.rename(old_preview, new_preview)

    # Create a new zip
    zip_path = os.path.join(RESULT_FOLDER, f"{session_id}_renamed.zip")
    with ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(result_dir):
            for file in files:
                filepath = os.path.join(root, file)
                zipf.write(filepath, os.path.relpath(filepath, result_dir))

    return send_file(zip_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
