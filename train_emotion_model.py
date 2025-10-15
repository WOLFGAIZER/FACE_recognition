# train_emotion_model.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from shufflenet_model import build_shufflenetv2

# ========== CONFIG ==========
IMAGE_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 25
DATASET_DIR = "datasets/fer2013"
MODEL_SAVE_PATH = "emotion_weights.h5"
NUM_CLASSES = 6  # angry, happy, sad, surprised, drowsy, fatigue

# ========== DATA AUGMENTATION ==========
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset='validation'
)

# ========== MODEL ==========
model = build_shufflenetv2(input_shape=(48, 48, 1), num_classes=NUM_CLASSES)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ========== CALLBACKS ==========
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ========== TRAIN ==========
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# ========== SAVE ==========
model.save(MODEL_SAVE_PATH)
print(f"✅ Model training complete. Saved at {MODEL_SAVE_PATH}")

