from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import numpy as np
from werkzeug.utils import secure_filename
import json
import sys
from datetime import datetime

# Create Flask app
app = Flask(__name__)
app.secret_key = 'brain_tumor_detection_secret_key'

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ===== ML MODEL INTEGRATION =====
def predict_tumor(image_path):
    """
    This is a placeholder function. Replace with your actual ML model prediction code.
    """
    tumor_types = ["glioma", "meningioma", "pituitary", "no_tumor"]
    import random
    prediction = random.choice(tumor_types)
    confidence = round(random.uniform(0.7, 0.99), 4)
    probabilities = {tumor: round(random.uniform(0, 1), 4) for tumor in tumor_types}
    probabilities[prediction] = confidence
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities
    }

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image with ML model
            result = predict_tumor(filepath)
            
            # Add the image path to the result
            result['image_path'] = f'uploads/{filename}'
            
            # Redirect to results page with the result data
            return redirect(url_for('results', result=json.dumps(result)))
        else:
            flash('File type not allowed. Please upload a JPG, JPEG, or PNG file.')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/results')
def results():
    result_json = request.args.get('result', '{}')
    try:
        result = json.loads(result_json)
    except json.JSONDecodeError:
        result = {}

    # Ensure the image path is included in the result
    return render_template('results.html', result=result)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Handle login logic here
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Placeholder logic for authentication (replace with actual logic)
        if username == 'admin' and password == 'password':
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
            return redirect(request.url)
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Handle signup logic here
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Placeholder logic for signup (replace with actual logic)
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(request.url)
        
        # Save user to database (placeholder)
        flash('Signup successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

if __name__ == '__main__':
    app.run(debug=True)