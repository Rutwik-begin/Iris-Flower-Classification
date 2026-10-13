
import json
import joblib
import gradio as gr
import pandas as pd

with open("artifacts/model_metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)

model = joblib.load("artifacts/iris_model.joblib")
feature_names = metadata["feature_names"]
target_names = metadata["target_names"]


def predict(sepal_length, sepal_width, petal_length, petal_width):
    sample = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=feature_names,
    )
    probabilities = model.predict_proba(sample)[0]
    return {
        target_names[i]: float(probabilities[i])
        for i in range(len(target_names))
    }


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(4.0, 8.0, value=5.1, label="Sepal length (cm)"),
        gr.Slider(2.0, 4.5, value=3.5, label="Sepal width (cm)"),
        gr.Slider(1.0, 7.0, value=1.4, label="Petal length (cm)"),
        gr.Slider(0.1, 2.5, value=0.2, label="Petal width (cm)"),
    ],
    outputs=gr.Label(num_top_classes=3, label="Predicted species"),
    title="Iris Flower Classification",
    description="Adjust the four flower measurements to predict its species.",
)

if __name__ == "__main__":
    demo.launch()
