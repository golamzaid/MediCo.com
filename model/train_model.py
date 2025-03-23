import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_model(input_shape=(224, 224, 3), num_classes=4):
    """
    Create a CNN model for brain tumor classification
    """
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Fully connected layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """
    Train the brain tumor classification model
    """
    # ===== ADD YOUR DATASET PATH HERE =====
    # Set the path to your dataset directory
    # The dataset should have the following structure:
    # dataset/
    #   ├── glioma/
    #   ├── meningioma/
    #   ├── no_tumor/
    #   └── pituitary/
    
 
    data_dir = r"C:\Users\golam\OneDrive\Desktop\CODES\PROJECTS\MRI\brain_tumor_flask\Training"

   
    # Set model save path
    model_save_path = "model/brain_tumor_model.h5"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Image parameters
    img_size = (224, 224)
    batch_size = 32
    epochs = 50
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Only rescaling for validation
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    # Load validation data
    validation_generator = valid_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Get class names and indices
    class_indices = train_generator.class_indices
    class_names = list(class_indices.keys())
    print(f"Classes: {class_names}")
    
    # Create the model
    model = create_model(input_shape=(*img_size, 3), num_classes=len(class_names))
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Save the final model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Save class indices
    import json
    with open(os.path.join(os.path.dirname(model_save_path), 'class_indices.json'), 'w') as f:
        json.dump(class_indices, f)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(model_save_path), 'training_history.png'))
    plt.show()
    
    return model, history, class_names

if __name__ == "__main__":
    train_model()