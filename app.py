import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris

# --- Load dataset ---
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# --- Load pre-trained model ---
model = tf.keras.models.load_model("iris_model.h5")

# --- Setup label encoder ---
le = LabelEncoder()
le.fit(class_names)

# --- Prediction function ---
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    # Validate inputs
    if not (4.3 <= sepal_length <= 7.9):
        return "❌ Sepal Length must be between 4.3 and 7.9 cm"
    if not (2.0 <= sepal_width <= 4.4):
        return "❌ Sepal Width must be between 2.0 and 4.4 cm"
    if not (1.0 <= petal_length <= 6.9):
        return "❌ Petal Length must be between 1.0 and 6.9 cm"
    if not (0.1 <= petal_width <= 2.5):
        return "❌ Petal Width must be between 0.1 and 2.5 cm"
    
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    preds = model.predict(input_data, verbose=0)
    class_index = np.argmax(preds, axis=1)[0]
    class_label = le.inverse_transform([class_index])[0]
    confidence = float(np.max(preds)) * 100
    return f"🌸 Predicted Species: {class_label}\n✅ Confidence: {confidence:.2f}%"

# --- Build Gradio interface with ranges in labels ---
with gr.Blocks() as demo:
    gr.Markdown("# 🌼 Iris Flower Classifier")
    gr.Markdown("Enter flower measurements below to predict the species.")

    with gr.Row():
        sepal_length = gr.Number(label="Sepal Length (cm) [4.3 - 7.9]", value=5.1, precision=1)
        sepal_width  = gr.Number(label="Sepal Width (cm) [2.0 - 4.4]", value=3.5, precision=1)
    with gr.Row():
        petal_length = gr.Number(label="Petal Length (cm) [1.0 - 6.9]", value=1.4, precision=1)
        petal_width  = gr.Number(label="Petal Width (cm) [0.1 - 2.5]", value=0.2, precision=1)
    
    predict_btn = gr.Button("🔮 Predict")
    output = gr.Textbox(label="Result", lines=2)
    
    predict_btn.click(
        fn=predict_iris,
        inputs=[sepal_length, sepal_width, petal_length, petal_width],
        outputs=output
    )

# --- Run the app ---
if __name__ == "__main__":
    demo.launch()
