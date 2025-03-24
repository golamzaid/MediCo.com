import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess an image for prediction."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def generate_gradcam(model, img_array, layer_name='conv2d_5'):
    """Generate Grad-CAM heatmap to highlight affected areas."""
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions)]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)[0]
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def overlay_heatmap(img_path, heatmap, alpha=0.5):
    """Overlay Grad-CAM heatmap onto original image."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    output_path = "gradcam_output.jpg"
    cv2.imwrite(output_path, superimposed_img)
    return output_path

def predict_tumor(img_path):
    """Predict tumor type and generate affected area visualization."""
    model_path = "model/brain_tumor_model.h5"
    class_indices_path = "model/class_indices.json"
    
    if not os.path.exists(model_path) or not os.path.exists(class_indices_path):
        return {"error": "Model or class indices not found. Train the model first."}
    
    model = load_model(model_path)
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    class_names = {v: k for k, v in class_indices.items()}
    
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)[0]
    
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]
    confidence = float(predictions[predicted_class_idx])
    probabilities = {class_names[i]: float(prob) for i, prob in enumerate(predictions)}
    
    heatmap = generate_gradcam(model, img_array)
    heatmap_path = overlay_heatmap(img_path, heatmap)
    
    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "heatmap": heatmap_path
    }

def predict_tumor_model(image_path):
    """
    Simulates tumor prediction. Replace this with your actual ML model logic.
    """
    tumor_types = ["glioma", "meningioma", "pituitary", "no_tumor"]
    prediction = random.choice(tumor_types)
    confidence = round(random.uniform(0.7, 0.99), 4)
    probabilities = {tumor: round(random.uniform(0, 1), 4) for tumor in tumor_types}
    probabilities[prediction] = confidence
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    result = predict_tumor(img_path)
    print(json.dumps(result, indent=2))
