from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Input, Dropout,
                                     RandomFlip, RandomRotation, RandomZoom)
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ── Data ──────────────────────────────────────────────────────────────────────
train_data = image_dataset_from_directory(
    'data/train',
    image_size=(128, 128),
    batch_size=32,
    label_mode='binary'
)

val_data = image_dataset_from_directory(
    'data/val',
    image_size=(128, 128),
    batch_size=32,
    label_mode='binary'
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = Sequential([
    Input(shape=(128, 128, 3)),

    # Data Augmentation (only active during training)
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),

    # Conv Block 1
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Conv Block 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Conv Block 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Classifier Head
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),              # kills 50% of neurons → fights overfitting
    Dense(1, activation='sigmoid')
])

# ── Compile ───────────────────────────────────────────────────────────────────
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# ── Callbacks ────────────────────────────────────────────────────────────────
callbacks = [
    # Stop training if val_loss doesn't improve for 3 consecutive epochs
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),

    # Save the best model automatically (lowest val_loss)
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
]

# ── Train ─────────────────────────────────────────────────────────────────────
model.fit(
    train_data,
    epochs=20,
    validation_data=val_data,
    callbacks=callbacks
)
