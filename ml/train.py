import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
from ml.augment import oversample_minority

# ── Reproducibility ───────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

# ── Load data ─────────────────────────────────────────────────────
print("Loading dataset...")
X_train = np.load('ml/data/X_train.npy')
y_train = np.load('ml/data/y_train.npy')

# ── Augment minority classes to 500 samples each ──────────────────
print("Augmenting minority classes...")
X_train, y_train = oversample_minority(X_train, y_train, target_per_class=500)
print(f"After augmentation: {X_train.shape[0]} training beats")
X_val   = np.load('ml/data/X_val.npy')
y_val   = np.load('ml/data/y_val.npy')

print(f"Train: {X_train.shape} | Val: {X_val.shape}")

# ── Reshape for CNN — (samples, timesteps, channels) ─────────────
X_train = X_train[..., np.newaxis]
X_val   = X_val[..., np.newaxis]

# ── One-hot encode labels ─────────────────────────────────────────
num_classes = 5
y_train_oh  = keras.utils.to_categorical(y_train, num_classes)
y_val_oh    = keras.utils.to_categorical(y_val,   num_classes)

# ── Class weights (fixes imbalance) ──────────────────────────────
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_classes),
    y=y_train
)
class_weight_dict = dict(enumerate(weights))
print(f"Class weights: {class_weight_dict}")

# ── Build model ───────────────────────────────────────────────────
def build_model(input_shape=(180, 1), num_classes=5):
    inputs = keras.Input(shape=input_shape)

    # Block 1
    x = keras.layers.Conv1D(32, 5, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)

    # Block 2
    x = keras.layers.Conv1D(64, 5, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)

    # Block 3
    x = keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)

    # Block 4
    x = keras.layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    # Head
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs)


model = build_model()
model.summary()

# ── Compile ───────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ── Callbacks ─────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='models/arrhythmia_cnn_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    ),
]

# ── Train ─────────────────────────────────────────────────────────
print("\nStarting training...")
history = model.fit(
    X_train, y_train_oh,
    validation_data=(X_val, y_val_oh),
    epochs=100,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ── Save final model ──────────────────────────────────────────────
model.save('models/arrhythmia_cnn_final.h5')
print("\nFinal model saved to models/arrhythmia_cnn_final.h5")

# ── Results ───────────────────────────────────────────────────────
best_val_acc = max(history.history['val_accuracy'])
best_epoch   = history.history['val_accuracy'].index(best_val_acc) + 1

print(f"\n--- Training Complete ---")
print(f"Best val accuracy : {best_val_acc * 100:.2f}%")
print(f"Best epoch        : {best_epoch}")
print(f"Target            : > 90%")

if best_val_acc >= 0.90:
    print("TARGET MET ✅")
else:
    print("Below target — see evaluate.py for next steps")