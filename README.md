

# Traffic Sign Recognition Using CNN and Streamlit

This project demonstrates a Convolutional Neural Network (CNN) model trained to recognize 43 different types of traffic signs. The model is deployed using a **Streamlit web app**, and it can also be tested in **real-time using a webcam feed**.

## Demo

- **Streamlit Web App**: Upload an image of a traffic sign and get predictions with confidence scores and a probability bar chart.
- **Real-Time Webcam Testing**: Detect and classify traffic signs live from your laptop camera.

---

## Contents

- `traffic_classifier.h5`: Trained CNN model
- `streamlit_app.py`: Streamlit web application
- `realtime_test.py`: Script to run live webcam recognition
- `train_model.ipynb`: Notebook for model training
- `Test.csv`: Test dataset for evaluation
- `trafic_sign/train/`: Training images, organized by class

---

## Model Accuracy

- **Training Accuracy**: ~98%
- **Test Accuracy**: ~96.66%

---

## Installation

```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
