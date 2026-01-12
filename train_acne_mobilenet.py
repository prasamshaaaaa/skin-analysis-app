import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# ----------------- Dataset -----------------
data_dir = "dataset"  # Your dataset folder (acne/clear)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

# ----------------- MobileNetV2 Base -----------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)  # 1 output: acne or clear

model = Model(inputs=base_model.input, outputs=x)

# Compile
model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# ----------------- Train -----------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# ----------------- Save Model -----------------
model.save("acne_model.h5")
print("Training complete! Model saved as acne_model.h5")
