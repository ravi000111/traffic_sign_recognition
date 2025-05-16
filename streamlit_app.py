import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the model once
@st.cache_resource
def load_model(path='traffic_classifier.h5'):
    model = tf.keras.models.load_model(path)
    return model

# Hard-coded mapping from class IDs to sign names
class_id_to_name = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

model = load_model()

st.title("🚦 Traffic Sign Recognition")
st.write("Upload an image of a traffic sign, and the model will predict its class and show detailed probabilities.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file:
    # Load and preprocess image exactly as during training
    image = Image.open(uploaded_file)
    # Resize
    image = image.resize((30, 30))
    # Convert to numpy array
    img_array = np.array(image)
    # Drop alpha channel if present
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    # Normalize
    img_array = img_array / 255.0
    # Add batch dimension
    img_array = img_array.reshape(1, 30, 30, 3)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict
    preds = model.predict(img_array).flatten()
    # Display raw probabilities
    st.write("**Raw model probabilities:**")
    st.write(np.round(preds, 4))

    # Top-1 prediction
    top1 = int(np.argmax(preds))
    st.markdown(f"**Predicted Class Index:** {top1}")
    st.markdown(f"**Predicted Sign Name:** {class_id_to_name[top1]}")
    st.markdown(f"**Confidence:** {preds[top1]*100:.2f}%")

    # Top-5 predictions
    top5_idx = preds.argsort()[-5:][::-1]
    st.write("**Top-5 Predictions:**")
    for idx in top5_idx:
        st.write(f"- {class_id_to_name[idx]}: {preds[idx]*100:.2f}%")

    # Optional: probability bar chart
    import pandas as pd
    df_probs = pd.DataFrame([preds], columns=[class_id_to_name[i] for i in range(len(preds))])
    st.bar_chart(df_probs)
