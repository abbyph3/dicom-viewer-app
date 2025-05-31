import streamlit as st
import pydicom
import numpy as np
import cv2
import os
import json

def load_dicom_image(file_path):
    dicom = pydicom.dcmread(file_path)
    image = dicom.pixel_array.astype(np.float32)
    image -= np.min(image)
    if np.max(image) != 0:
        image /= np.max(image)
    image *= 255.0
    return image.astype(np.uint8), dicom

def apply_brightness_contrast(image, brightness=0, contrast=1.0):
    image = image.astype(np.float32)
    image = image * contrast + brightness
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)

def apply_filter(image, filter_type):
    if filter_type == "Gaussian Blur":
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == "Sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == "Edge Detection":
        return cv2.Canny(image, 100, 200)
    else:
        return image

def get_filter_explanation(filter_type):
    explanations = {
        "Original": "No filter applied; view the raw image.",
        "Gaussian Blur": "Smooths the image to reduce noise and detail, useful for preprocessing.",
        "Sharpen": "Enhances edges and details to make structures more distinct.",
        "Edge Detection": "Detects boundaries using Canny algorithm, highlighting edges."
    }
    return explanations.get(filter_type, "")

def dicom_to_dict_str(dicom):
    dicom_dict = {}
    for elem in dicom:
        if elem.VR != "SQ":
            tag_name = elem.keyword if elem.keyword else str(elem.tag)
            dicom_dict[tag_name] = str(elem.value)
    return json.dumps(dicom_dict, indent=4)

def zoom_image(image, zoom_factor):
    h, w = image.shape
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    cropped = image[top:top+new_h, left:left+new_w]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return zoomed

def load_multi_dicom(uploaded_files):
    dicoms = []
    for uploaded_file in uploaded_files:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            dicom = pydicom.dcmread(temp_path)
            dicoms.append((dicom, temp_path))
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
            os.remove(temp_path)
            continue

    def sort_key(x):
        dicom = x[0]
        return getattr(dicom, "InstanceNumber", 0) or getattr(dicom, "SliceLocation", 0) or 0

    dicoms.sort(key=sort_key)

    images = []
    for dicom, path in dicoms:
        img = dicom.pixel_array.astype(np.float32)
        img -= np.min(img)
        if np.max(img) != 0:
            img /= np.max(img)
        img *= 255.0
        images.append(img.astype(np.uint8))
    return images, dicoms

def extract_searchable_info(dicom):
    return {
        "PatientName": str(getattr(dicom, "PatientName", "")),
        "PatientID": str(getattr(dicom, "PatientID", "")),
        "StudyDate": str(getattr(dicom, "StudyDate", "")),
        "Modality": str(getattr(dicom, "Modality", "")),
        "SliceThickness": str(getattr(dicom, "SliceThickness", "")),
        "PixelSpacing": str(getattr(dicom, "PixelSpacing", ""))
    }

def main():
    st.set_page_config(page_title="Advanced DICOM Viewer", layout="centered")
    st.title("🩻 Advanced DICOM Viewer & Processor")

    st.sidebar.header("Upload DICOM Files")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more DICOM files",
        type=["dcm"],
        accept_multiple_files=True
    )

    if uploaded_files:
        images, dicoms = load_multi_dicom(uploaded_files)
        if not images:
            st.error("No valid DICOM images loaded.")
            return

        # Multi-slice navigation if more than one image
        slice_idx = 0
        if len(images) > 1:
            slice_idx = st.sidebar.slider("Slice", 0, len(images)-1, 0)

        original_image = images[slice_idx]
        current_dicom = dicoms[slice_idx][0]

        # --- Dropdown for basic metadata fields ---
        st.subheader("📄 Selected Metadata Viewer")
        metadata_fields = [
            "PatientName",
            "PatientID",
            "StudyDate",
            "Modality",
            "SliceThickness",
            "PixelSpacing"
        ]
        selected_field = st.selectbox("Choose metadata field to view", metadata_fields)
        field_value = getattr(current_dicom, selected_field, "Not Found")
        st.write(f"**{selected_field}**: {field_value}")

        # --- Full metadata inside collapsible tab ---
        with st.expander("📁 Show full DICOM Metadata"):
            metadata_str = dicom_to_dict_str(current_dicom)
            st.code(metadata_str, language="json")

        # --- Image Processing Controls ---
        st.subheader("🛠️ Brightness / Contrast / Filter Controls")
        col1, col2 = st.columns(2)
        with col1:
            brightness = st.slider("Brightness", -100, 100, 0)
        with col2:
            contrast = st.slider("Contrast", 0.1, 3.0, 1.0)

        filter_type = st.selectbox(
            "Filter",
            ["Original", "Gaussian Blur", "Sharpen", "Edge Detection"]
        )
        explanation = get_filter_explanation(filter_type)
        st.markdown(f"**Filter Explanation:** {explanation}")

        zoom_factor = st.slider("Zoom", 1.0, 5.0, 1.0, 0.1)

        # --- Process and display image ---
        processed = apply_brightness_contrast(original_image, brightness, contrast)
        processed = apply_filter(processed, filter_type)
        processed = zoom_image(processed, zoom_factor)

        if filter_type == "Edge Detection":
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

        st.image(
            processed,
            caption=f"Slice {slice_idx+1} | Filter: {filter_type} | Zoom: {zoom_factor}x",
            use_container_width=True,
            clamp=True
        )

        # --- Clean up temp files ---
        for _, path in dicoms:
            if os.path.exists(path):
                os.remove(path)

    else:
        st.info("Please upload one or more DICOM files (.dcm) to begin.")

if __name__ == "__main__":
    main()
