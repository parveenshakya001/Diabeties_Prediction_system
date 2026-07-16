# app.py

import os
import gradio as gr
import joblib

# We load the model once when the app starts
deployed_lr = joblib.load('diabetes_prediction_model.pkl')

# --- CODE BLOCK: REMOVED SPACES DECORATOR ---
def predict_rent(size_of_prop):
    # The model expects a 2D array: [[size]]
    prediction = deployed_lr.predict([[size_of_prop]])
    # Extract the single prediction value and format it
    return f"Estimated Rent: {prediction[0]:.2f}"
# --------------------------------------------

interface = gr.Interface(
    fn=predict_rent,
    inputs=gr.Number(label="Please Enter the Size of Your Property for rent"),
    outputs=gr.Text(label="Predicted Rent"),
    title="Property Rent Predictor",
    description="Enter the property size to get a rent estimate powered by Machine Learning."
)

if __name__ == "__main__":
    # --- CODE BLOCK: UPDATED LAUNCH PARAMETERS FOR RENDER ---
    interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
    # --------------------------------------------------------
