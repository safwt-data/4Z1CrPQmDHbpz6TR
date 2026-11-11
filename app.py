import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocessing




model = load_model("../models/mobilenetv2_tuned_model.keras",
                    custom_objects={ "preprocess_input": mobilenetv2_preprocessing},
                    compile = False,)

class_names = ["flip","nonflip"]

def predict_image(image):
    # `image` comes from the Gradio upload box.
    image = image.convert("RGB").resize((224, 224))
    batch = np.asarray(image, dtype=np.float32)[None, ...]
    score = float(model.predict(batch, verbose=0)[0, 0])
    return class_names[int(score >= 0.5)]

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Textbox(label="Prediction") )

if __name__ == "__main__":
    demo.launch()