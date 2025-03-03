import os
import tempfile
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pywt
from skimage.measure import shannon_entropy
from skimage import io, color
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from streamlit_cropper import st_cropper
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = load_model('model_final.keras')

# Function for read original image
def read_original_image(image_path):
    image_data_og = np.frombuffer(uploaded_file.read(), np.uint8)
    image_og = cv2.imdecode(image_data_og, cv2.IMREAD_COLOR)
    image_og_rgb = cv2.cvtColor(image_og, cv2.COLOR_BGR2RGB)

    return image_og_rgb

# Function for preprocessing the image
def preprocess_image(image):
    # Read the image
    # image_data = np.frombuffer(uploaded_file.read(), np.uint8)
    # image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    
    # Resize image to 128 x 128
    resized_image = cv2.resize(image, (128, 128))

    # Convert to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Adjust contrast of the image
    contrasted_image = cv2.convertScaleAbs(blurred_gray_image, alpha=1.5, beta=0)

    # Calculate gradient using Sobel operator
    grad_x = cv2.Sobel(contrasted_image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal gradient
    grad_y = cv2.Sobel(contrasted_image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical gradient

    # Calculate gradient magnitude
    grad_magnitude = cv2.magnitude(grad_x, grad_y)

    # Normalize gradient magnitude to [0, 255]
    grad_magnitude_normalized = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    grad_magnitude_normalized = np.uint8(grad_magnitude_normalized)

    # Apply Otsu's threshold to create binary mask
    _, binary_mask = cv2.threshold(grad_magnitude_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological closing to remove small holes
    elliptical_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, elliptical_kernel)

    # Apply dilation to enlarge white areas
    dilated_mask = cv2.dilate(closed_mask, elliptical_kernel, iterations=1)
    inverse_dilated_mask = cv2.bitwise_not(dilated_mask)
    non_black_pixels = np.sum(dilated_mask > 0)
    gray_image_3d = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    # Apply mask to original image
    image_masked = cv2.bitwise_and(resized_image, resized_image, mask=dilated_mask)
    image_masked_inverse = cv2.bitwise_and(gray_image_3d, gray_image_3d, mask=inverse_dilated_mask)
    image_masked_visual = cv2.add(image_masked, image_masked_inverse)
    return image_masked_visual, image_masked, non_black_pixels

# Function to extract features using DWT and calculate energy and entropy
def extract_features(output_path):
    image = io.imread(output_path)
    image = color.rgb2gray(image)
    coeffs = pywt.dwt2(image, 'haar')  # You can use other wavelets like 'db1', 'sym2', etc.
    cA, (cH, cV, cD) = coeffs  # Approximation, Horizontal, Vertical, Diagonal

    def calculate_features(sub_band):
        energy = np.sum(np.square(sub_band))
        entropy = shannon_entropy(sub_band)
        return energy, entropy

    # Extract features from each sub-band
    features = {}
    features['cA_energy'], features['cA_entropy'] = calculate_features(cA)
    features['cH_energy'], features['cH_entropy'] = calculate_features(cH)
    features['cV_energy'], features['cV_entropy'] = calculate_features(cV)
    features['cD_energy'], features['cD_entropy'] = calculate_features(cD)

    return features

# Function to identify severity based on the ratio of colored pixels
def identify_severity(image,non_black_pixels):
    total_pixels = image.shape[0] * image.shape[1]
    colored_pixels = np.count_nonzero(image)

    # Calculate the ratio of colored pixels
    ratio = non_black_pixels / 16384 * 100

    # Define severity categories based on the ratio
    if ratio == 0:
        severity = "No colored pixels"
    elif ratio < 20:
        severity = "Light"
    elif ratio < 50:
        severity = "Moderate"
    elif ratio < 80:
        severity = "Moderate to Severe"
    else:
        severity = "Severe"

    return severity, ratio

def predict_input(features_df):
    scaler = StandardScaler()

    train_data = pd.read_csv('train_data.csv')

    X_train = train_data.drop(columns=['target', 'filename'])

    X_train_normalized = scaler.fit_transform(X_train)
    # Pastikan X_new memiliki bentuk 2D untuk normalisasi
    # X_new_values = np.array(list(X_new.values()))
    
    # X_new_reshaped = X_new_values.reshape(1, -1)

    X_new = features_df.iloc[[0]]
    # Transform data baru menggunakan scaler yang sudah difit pada data training
    X_new_normalized = scaler.transform(X_new)

    # Lakukan prediksi
    prediction = model.predict(X_new_normalized)
    pred_labels = (prediction > 0.5).astype(int) 
    
    if pred_labels == 1:
        label = 'Monkeypox'
    else:
        label = 'Non-Monkeypox'

    # Return hasil prediksi
    return label

# Function for deleting temporary files
def delete_temp_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

# Streamlit UI
st.title("MOTION : Monkeypox Identification")

# File uploader for image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the original image
    original_image = read_original_image(uploaded_file)
    processed_image = None
    # Opsi apakah pengguna ingin melakukan crop
    use_cropper = st.checkbox("Crop image before processing?", value=False)

    if use_cropper:
        pil_image = Image.fromarray(original_image)
        cropped_image = st_cropper(pil_image, aspect_ratio=(1, 1), return_type="image")
        # Simpan gambar yang di-crop
        if st.button("Done Selecting"):
            try:
        # Mengonversi gambar cropped menjadi format yang sesuai untuk OpenCV
                cropped_image_cv_1 = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)
                cropped_image_cv_2 = cv2.cvtColor(np.array(cropped_image_cv_1), cv2.COLOR_BGR2RGB)
                processed_image = cropped_image_cv_2
             # Gambar yang diproses adalah gambar yang di-crop
            except Exception as e:
                # st.error("An error occurred while processing the cropped image.")
                # Debugging: Uncomment the line below to see the error in console
                # print(e)
                processed_image = None  # Set processed_image to None if there's an error
    else:
        try:
            processed_image = original_image 
        except Exception as e:
            # st.error("An error occurred while processing the original image.")
            # Debugging: Uncomment the line below to see the error in console
            # print(e)
            processed_image = None  

    # Preprocess the image
    if processed_image is not None:
        visual_image, preprocessed_image, non_black_pixels = preprocess_image(processed_image)

        # Save preprocessed image to a temporary file'
        temp_dir = tempfile.gettempdir()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=temp_dir) as temp_file:
            temp_file_path = temp_file.name
            cv2.imwrite(temp_file_path, preprocessed_image)

        try:
            # Extract features
            features = extract_features(temp_file_path)
            features_df = pd.DataFrame([features])

            # Make a prediction
            prediction = predict_input(features_df)

            # Display results
            severity = None
            ratio = 0
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("### Original Image")
                st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), channels="BGR", caption="Original Image", use_column_width=True)

            with col2:
                st.write("### Preprocessed Image")
                st.image(cv2.cvtColor(visual_image, cv2.COLOR_BGR2RGB), channels="BGR", caption="Preprocessed Image", use_column_width=True)

            with col3:
                st.write("### Prediction Result")
                st.write(f"Prediction: {prediction}")
                if prediction == "Monkeypox":
                    severity, ratio = identify_severity(preprocessed_image, non_black_pixels)
                    st.write(f"Severity: {severity}, {ratio}%")

        finally:
            # Ensure temporary file is deleted
            delete_temp_file(temp_file_path)
