"""
Body Type Classification Model Training Script
This is a placeholder script for training a CNN model to classify body types.
In a real implementation, you would need a dataset of body images with labels.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

def create_simple_cnn_model():
    """
    Create a simple CNN model for body type classification
    This is a placeholder model - in reality, you'd need proper training data
    """
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(224, 224, 3)),
        
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')  # 3 body types: Ectomorph, Mesomorph, Endomorph
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_dummy_dataset():
    """
    Create dummy data for demonstration purposes
    In reality, you would load your actual dataset here
    """
    # Generate dummy images (random noise)
    num_samples = 1000
    X_train = np.random.rand(num_samples, 224, 224, 3)
    X_val = np.random.rand(200, 224, 224, 3)
    
    # Generate dummy labels (random distribution)
    y_train = np.random.randint(0, 3, num_samples)
    y_val = np.random.randint(0, 3, 200)
    
    # Convert to categorical
    y_train = tf.keras.utils.to_categorical(y_train, 3)
    y_val = tf.keras.utils.to_categorical(y_val, 3)
    
    return X_train, y_train, X_val, y_val

def train_model():
    """
    Train the body type classification model
    """
    print("Creating model...")
    model = create_simple_cnn_model()
    
    print("Creating dummy dataset...")
    X_train, y_train, X_val, y_val = create_dummy_dataset()
    
    print("Training model...")
    # Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        keras.callbacks.ModelCheckpoint(
            'models/body_type_model_best.h5',
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.save('models/body_type_model.h5')
    print("Model saved to models/body_type_model.h5")
    
    return model, history

def load_trained_model():
    """
    Load the trained model
    """
    try:
        model = keras.models.load_model('models/body_type_model.h5')
        print("Model loaded successfully!")
        return model
    except:
        print("No trained model found. Creating a new one...")
        return None

def predict_body_type(model, image_path):
    """
    Predict body type from an image
    """
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    
    # Make prediction
    predictions = model.predict(img_array)
    body_types = ['Ectomorph', 'Mesomorph', 'Endomorph']
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    return body_types[predicted_class], confidence

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Try to load existing model, otherwise train new one
    model = load_trained_model()
    
    if model is None:
        print("Training new model...")
        model, history = train_model()
        print("Training completed!")
    
    # Test prediction (if you have a test image)
    # test_image_path = "path/to/test/image.jpg"
    # if os.path.exists(test_image_path):
    #     body_type, confidence = predict_body_type(model, test_image_path)
    #     print(f"Predicted body type: {body_type} (confidence: {confidence:.2f})") 