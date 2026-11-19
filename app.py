from contextlib import contextmanager

import pandas as pd
import streamlit as st
from pycaret.regression import load_model, predict_model

model = load_model("gbr_insurance_charges")

LOADING_OVERLAY_HTML = """
<style>
.loading-overlay {
    position: fixed;
    inset: 0;
    z-index: 999999;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    background: rgba(0, 0, 0, 0.45);
    color: #fff;
    font-size: 1.1rem;
}
.loading-overlay .spinner {
    width: 48px;
    height: 48px;
    border: 5px solid rgba(255, 255, 255, 0.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: loading-spin 0.8s linear infinite;
}
@keyframes loading-spin {
    to { transform: rotate(360deg); }
}
</style>
<div class="loading-overlay">
    <div class="spinner"></div>
    <div>Predicting...</div>
</div>
"""


@contextmanager
def loading_overlay():
    """Cover the page with a loading overlay while the block runs."""
    placeholder = st.empty()
    placeholder.markdown(LOADING_OVERLAY_HTML, unsafe_allow_html=True)
    try:
        yield
    finally:
        placeholder.empty()

def predict(model, input_df):
    predictions_df = predict_model(estimator = model, data = input_df)
    predictions = predictions_df.iloc[0]['prediction_label']
    return predictions

def run():
    from PIL import Image

    image = Image.open("logo.png")
    image_hospital = Image.open("hospital.jpg")
    st.image(image, width="content")
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", ("Online", "Batch")
    )

    st.sidebar.info("This app is created to predict patient hospital charges")
    st.sidebar.success("https://www.pycaret.org")

    st.sidebar.image(image_hospital)
    st.title("Insurance Charges Prediction App")

    if add_selectbox == "Online":
        age = st.number_input('Age', min_value = 1, max_value = 100, value = 25)
        sex = st.selectbox('Sex', ['male', 'female'])
        bmi = st.number_input('BMI', min_value = 10, max_value = 100, value = 25)
        children = st.selectbox('Children', [0,1,2,3,4,5,6,7,8,9,10])
        if st.checkbox("Smoker"):
            smoker = 'yes'
        else:
            smoker = 'no'
        region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])

        output = ""
        input_dict = {
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region,
        }
        input_df = pd.DataFrame([input_dict])

        if st.button('Predict'):
            with loading_overlay():
                output = predict(model = model, input_df = input_df)
            output = '$' + str(output)
            st.success(f"The output is {output}")

    elif add_selectbox == "Batch":
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            with loading_overlay():
                predictions = predict_model(estimator=model, data=data)
            st.write(predictions)


if __name__ == "__main__":
    run()
