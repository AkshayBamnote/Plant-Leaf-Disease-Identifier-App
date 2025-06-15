import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Available GPUs: ", tf.config.list_physical_devices('GPU'))
import numpy as np
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Data generators
train_datagen = ImageDataGenerator(
    zoom_range=0.5,
    shear_range=0.3,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train = train_datagen.flow_from_directory(
    'D:/Plant Leaf disease detection Imgs/Archive (10)/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train',  # ✅ Replace with actual path
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

val = val_datagen.flow_from_directory(
    'D:/Plant Leaf disease detection Imgs/Archive (10)/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid',  # ✅ Replace with actual path
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

# Save class labels
with open("class_labels.json", "w") as f:
    json.dump({str(v): k for k, v in train.class_indices.items()}, f)

# Load VGG19 and freeze
base_model = VGG19(input_shape=(256, 256, 3), include_top=False)
for layer in base_model.layers:
    layer.trainable = False

# Classification head
X = Flatten()(base_model.output)
X = Dense(38, activation='softmax')(X)
model = Model(base_model.input, X)

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
training = model.fit(train, steps_per_epoch=16, epochs=50, validation_data=val, validation_steps=16)

# Evaluate
accuracy = model.evaluate(val)[1]
print(f"The Accuracy of your model is {accuracy*100:.2f}%")

# Save model
model.save("plant_disease_detector_vgg19.h5")
