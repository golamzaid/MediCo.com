import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_model(input_shape=(224, 224, 3), num_classes=4):
    """Create a CNN model for brain tumor classification."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(data_dir, model_save_path="model/brain_tumor_model.h5", img_size=(224, 224), batch_size=32, epochs=50):
    """Train the brain tumor classification model."""
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Data augmentation and preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=20, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode='nearest', validation_split=0.2
    )
    valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    # Data generators
    train_generator = train_datagen.flow_from_directory(
        data_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', subset='training'
    )
    validation_generator = valid_datagen.flow_from_directory(
        data_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', subset='validation'
    )
    
    # Get class names
    class_indices = train_generator.class_indices
    class_names = list(class_indices.keys())
    print(f"Classes: {class_names}")
    
    # Create model
    model = create_model(input_shape=(*img_size, 3), num_classes=len(class_names))
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    ]
    
    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Save model and class indices
    model.save(model_save_path)
    with open(os.path.join(os.path.dirname(model_save_path), 'class_indices.json'), 'w') as f:
        json.dump(class_indices, f)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(model_save_path), 'training_history.png'))
    plt.show()
    
    return model, history, class_names

if __name__ == "__main__":
    dataset_path = r"C:\Users\golam\OneDrive\Desktop\CODES\PROJECTS\MediCo.com\DataSets"
    train_model(dataset_path)
