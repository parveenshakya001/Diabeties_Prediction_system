import os
import gradio as gr
import joblib

# Load the trained model
try:
    deployed_dt = joblib.load("diabetes_prediction_model.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")


def predict_diabetes(pregnancies, glucose, insulin, bmi, age):
    try:
        # Convert inputs to numeric values
        input_data = [[
            float(pregnancies),
            float(glucose),
            float(insulin),
            float(bmi),
            float(age)
        ]]

        # Make prediction
        prediction = deployed_dt.predict(input_data)

        if prediction[0] == 1:
            return "Prediction: High Risk of Diabetes (Positive)"
        else:
            return "Prediction: Low Risk of Diabetes (Negative)"

    except Exception as e:
        return f"Prediction Error: {e}"


# Gradio Interface
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Age"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Diabetes Prediction System",
    description="Enter the patient details to predict diabetes risk."
)

if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
