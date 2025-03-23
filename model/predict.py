import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """
    Load and preprocess an image for prediction
    
    Args:
        img_path: Path to the image file
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image ready for model prediction
    """
    # Load image
    img = image.load_img(img_path, target_size=target_size)
    
    # Convert to array and normalize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

def predict_tumor(img_path):
    """
    Predict tumor type from brain MRI scan
    
    Args:
        img_path: Path to the input image
        
    Returns:
        Dictionary with prediction results
    """
    # ===== ADD YOUR MODEL PATH HERE =====
    # Path to your trained model
    model_path = "model/brain_tumor_model.h5"
    
    # Path to class indices JSON file
    class_indices_path = "model/class_indices.json"
    
    # Check if model exists
    if not os.path.exists(model_path):
        return {
            "error": "Model not found. Please train the model first."
        }
    
    # Check if class indices exist
    if not os.path.exists(class_indices_path):
        return {
            "error": "Class indices not found. Please train the model first."
        }
    
    # Load model
    model = load_model(model_path)
    
    # Load class indices
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    
    # Invert class indices to get class names
    class_names = {v: k for k, v in class_indices.items()}
    
    # Preprocess image
    img_array = load_and_preprocess_image(img_path)
    
    # Make prediction
    predictions = model.predict(img_array)[0]
    
    # Get predicted class and confidence
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]
    confidence = float(predictions[predicted_class_idx])
    
    # Create probabilities dictionary
    probabilities = {class_names[i]: float(prob) for i, prob in enumerate(predictions)}
    
    # Create result dictionary
    result = {
        "prediction": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities
    }
    
    return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    result = predict_tumor(img_path)
    
    print(json.dumps(result, indent=2))