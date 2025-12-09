import os
import io
import base64
import re
import time
import numpy as np
import webbrowser
from threading import Timer
from flask import Flask, render_template, request, jsonify, session
from pydub import AudioSegment
import requests
from sklearn.preprocessing import normalize
import uuid
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering

app = Flask(__name__)
app.secret_key = "pyannote_secret_key"

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

SERVER_STORE = {}

BASE_URL = "https://api.pyannote.ai/v1"

def get_headers(token):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def upload_to_pyannote(file_path, token):
    filename = os.path.basename(file_path)
    safe_name = re.sub(r'[^a-zA-Z0-9\-\.]', '-', filename)
    safe_name = re.sub(r'-+', '-', safe_name)
    
    print(f"DEBUG: Demande d'upload pour {safe_name}...")
    res = requests.post(f"{BASE_URL}/media/input", headers=get_headers(token), json={"url": f"media://{safe_name}"})
    if res.status_code != 201: raise Exception(f"API Error ({res.status_code}): {res.text}")
    
    data = res.json()
    with open(file_path, 'rb') as f:
        res_put = requests.put(data['url'], data=f, headers={"Content-Type": "audio/wav"})
    if res_put.status_code not in [200, 201]: raise Exception(f"S3 Error ({res_put.status_code})")
    
    return f"media://{safe_name}"

def decode_embedding(b64_str):
    return np.frombuffer(base64.b64decode(b64_str), dtype=np.float32)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'audio' not in request.files: return jsonify({"error": "No file part"}), 400
        file = request.files['audio']
        token = request.form.get('token')
        
        if not token: return jsonify({"error": "Token is missing"}), 400

        sid = str(uuid.uuid4())
        session['sid'] = sid 
        
        filepath = os.path.join(UPLOAD_FOLDER, f"{sid}_{file.filename}")
        file.save(filepath)
        
        media_key = upload_to_pyannote(filepath, token)
        
        SERVER_STORE[sid] = {
            "filepath": filepath,
            "media_key": media_key,
            "token": token,
            "embeddings": [] 
        }
        
        return jsonify({"status": "ok", "media_key": media_key})
        
    except Exception as e:
        print(f"CRASH UPLOAD: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/diarize', methods=['POST'])
def diarize():
    try:
        sid = session.get('sid')
        if not sid or sid not in SERVER_STORE: return jsonify({"error": "Session expired"}), 400
        user_data = SERVER_STORE[sid]
        
        res = requests.post(
            f"{BASE_URL}/diarize", 
            headers=get_headers(user_data['token']), 
            json={"url": user_data['media_key'], "model": "precision-2", "turnLevelConfidence": True}
        )
        if res.status_code != 200: return jsonify({"error": res.text}), 400

        job_id = res.json()['jobId']
        
        for _ in range(45):
            res = requests.get(f"{BASE_URL}/jobs/{job_id}", headers=get_headers(user_data['token']))
            if res.status_code == 200:
                status = res.json()['status']
                if status == 'succeeded': return jsonify(res.json()['output'])
                if status == 'failed': return jsonify({"error": "Diarization failed"}), 500
            time.sleep(2)
            
        return jsonify({"error": "Timeout"}), 504

    except Exception as e:
        print(f"CRASH DIARIZE: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/extract-segment', methods=['POST'])
def extract_segment():
    try:
        sid = session.get('sid')
        if not sid or sid not in SERVER_STORE: return jsonify({"error": "Session expired"}), 400
        
        user_data = SERVER_STORE[sid]
        data = request.json
        start = data.get('start')
        end = data.get('end')
        
        audio = AudioSegment.from_file(user_data['filepath'])
        cut = audio[int(start*1000):int(end*1000)]
        buf = io.BytesIO()
        cut.export(buf, format="wav")
        
        temp_name = f"temp-{uuid.uuid4()}.wav"
        res = requests.post(f"{BASE_URL}/media/input", headers=get_headers(user_data['token']), json={"url": f"media://{temp_name}"})
        if res.status_code != 201: raise Exception(res.text)
        
        requests.put(res.json()['url'], data=buf.getvalue(), headers={"Content-Type": "audio/wav"})
        
        res = requests.post(f"{BASE_URL}/voiceprint", headers=get_headers(user_data['token']), json={"url": f"media://{temp_name}"})
        if res.status_code != 200: raise Exception(res.text)
        
        vp_job = res.json()['jobId']
        
        embedding = None
        for _ in range(15):
            r = requests.get(f"{BASE_URL}/jobs/{vp_job}", headers=get_headers(user_data['token']))
            status = r.json()['status']
            if status == 'succeeded':
                embedding = r.json()['output']['voiceprint']
                break
            elif status == 'failed':
                return jsonify({"error": "No speech detected"}), 400
            time.sleep(1)
            
        if not embedding: return jsonify({"error": "Timeout"}), 504
        
        vector = decode_embedding(embedding).tolist()
        
        existing_idx = next((i for i, item in enumerate(user_data['embeddings']) if item["id"] == data.get('id')), -1)
        new_entry = {
            "id": data.get('id'),
            "speaker": data.get('speaker'),
            "vector": vector,
            "start": start,
            "end": end
        }
        
        if existing_idx >= 0:
            user_data['embeddings'][existing_idx] = new_entry
        else:
            user_data['embeddings'].append(new_entry)
        
        return jsonify({"status": "ok", "id": data.get('id')})
        
    except Exception as e:
        print(f"CRASH EXTRACT: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/compute-pca', methods=['GET'])
def compute_pca():
    try:
        sid = session.get('sid')
        if not sid or sid not in SERVER_STORE: return jsonify({"error": "Session expired"}), 400
        
        embs = SERVER_STORE[sid]['embeddings']
        if len(embs) < 3:
            return jsonify({"error": "Need at least 3 segments for t-SNE viz"}), 400
        
        X = np.array([e['vector'] for e in embs])
        
        X_norm = normalize(X, norm='l2')
        
        n_samples = len(X)
        perplexity = min(30, n_samples - 1)
        if perplexity < 1: perplexity = 1
        
        tsne = TSNE(n_components=2, perplexity=perplexity, init='random', learning_rate='auto', random_state=42)
        components = tsne.fit_transform(X_norm)
        
        results = []
        for i, e in enumerate(embs):
            results.append({
                "id": e['id'],
                "speaker": e['speaker'],
                "x": float(components[i][0]),
                "y": float(components[i][1]),
                "start": e['start'],
                "end": e['end']
            })
            
        return jsonify(results)
    except Exception as e:
        print(f"CRASH t-SNE: {e}")
        return jsonify({"error": str(e)}), 500

    
@app.route('/api/recluster', methods=['POST'])
def recluster():
    try:
        sid = session.get('sid')
        if not sid or sid not in SERVER_STORE: return jsonify({"error": "Session expired"}), 400
        
        embs = SERVER_STORE[sid]['embeddings']
        num_clusters = int(request.json.get('num_clusters', 2))
        
        if len(embs) < num_clusters:
            return jsonify({"error": f"Not enough points"}), 400
        
        X = np.array([e['vector'] for e in embs])
        
        X_norm = normalize(X, norm='l2')
        
        spectral = SpectralClustering(
            n_clusters=num_clusters,
            affinity='nearest_neighbors',
            n_neighbors=min(10, len(X)-1),
            random_state=42
        )
        labels = spectral.fit_predict(X_norm)
        
        mapping = {}
        for i, label_id in enumerate(labels):
            seg_id = embs[i]['id']
            mapping[seg_id] = f"New_Speaker_{label_id + 1}"
            embs[i]['speaker'] = mapping[seg_id]
            
        return jsonify(mapping)

    except Exception as e:
        print(f"CRASH CLUSTERING: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        def open_browser():
            webbrowser.open_new("http://127.0.0.1:5000")
        Timer(1, open_browser).start()
    
    app.run(debug=True, port=5000)