from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf
from matplotlib import colormaps
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# more flexible 
model_path = Path(__file__).resolve().parent / "models" / "mobilenetv2_tuned_model.keras"

model = keras.models.load_model(
    model_path,
    custom_objects={"preprocess_input": preprocess_input},
    compile=False,
)

class_names = ["flip", "nonflip"]  # same order in training

# Find MobileNetV2 inside the saved model for Grad-Cam as it needs info from the middle too
base_model = next(layer for layer in model.layers if isinstance(layer, keras.Model))
base_index = model.layers.index(base_model)


def predict_image(image):
    if image is None:
        raise gr.Error("Upload an image first.")

    # Prepare the uploaded image.
    original = image.convert("RGB")
    resized = original.resize((224, 224))
    # [None, ...] adds a dimension at the front. The 1 in (1, 224, 224, 3) 
    #  ... keeps all the existing dimensions.
    batch = tf.convert_to_tensor(np.asarray(resized)[None, ...], dtype=tf.float32)

    # Apply the saved preprocessing, then get MobileNetV2's feature maps.
    for layer in model.layers[:base_index]:
        batch = layer(batch, training=False)

    features = base_model(batch, training=False)

    # Get the prediction and track how it depends on the feature maps.
    with tf.GradientTape() as tape:
        tape.watch(features)
        prediction = features
        for layer in model.layers[base_index + 1:]:
            prediction = layer(prediction, training=False)

        probability = prediction[0, 0]
        class_index = int(probability.numpy() >= 0.5)
        score = probability if class_index == 1 else 1 - probability

    # Build the Grad-CAM heatmap.
    gradients = tape.gradient(score, features)
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(features[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))

    # Put the heatmap over the uploaded image.
    colors = colormaps["jet"](heatmap.numpy())[..., :3]
    overlay = Image.fromarray((colors * 255).astype(np.uint8))
    overlay = overlay.resize(original.size)
    overlay = Image.blend(original, overlay, 0.4)

    return class_names[class_index], overlay


if __name__ == "__main__":
    gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil", label="Upload an image"),
        outputs=[
            gr.Textbox(label="Prediction"),
            gr.Image(label="Grad-CAM"),
        ],
    ).launch()

# & "$env:TEMP\gradio-app\Scripts\python.exe" .\app.py 
# you need that as the current environment is using a long upload and this work 