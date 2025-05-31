import streamlit as st
import pydicom
import numpy as np
import cv2
import os
import json

def load_dicom_image(file_path):
    """
    Load DICOM file and return normalized 8-bit image array.
    """
    dicom = pydicom.dcmread(file_path)
    image = dicom.pixel_array.astype(np.float32)
    # Normalize pixel values to 0-255
    image -= np.min(image)
    if np.max(image) != 0:
        image /= np.max(image)
    image *= 255.0
    return image.astype(np.uint8), dicom

def apply_brightness_contrast(image, brightness=0, contrast=1.0):
    """
    Adjust brightness and contrast of image.
    Brightness: additive (-100 to 100)
    Contrast: multiplicative (0.1 to 3.0)
    """
    image = image.astype(np.float32)
    image = image * contrast + brightness
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)

def apply_filter(image, filter_type):
    """
    Apply selected filter to the image.
    """
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
    """
    Provide a brief explanation of each filter.
    """
    explanations = {
        "Original": "No filter applied; view the raw image.",
        "Gaussian Blur": "Smooths the image to reduce noise and detail, useful for preprocessing.",
        "Sharpen": "Enhances edges and details to make structures more distinct.",
        "Edge Detection": "Detects boundaries using Canny algorithm, highlighting edges."
    }
    return explanations.get(filter_type, "")

def dicom_to_dict_str(dicom):
    """
    Convert DICOM dataset to a dict with tag names and values,
    then return pretty JSON string.
    """
    dicom_dict = {}
    for elem in dicom:
        # Ignore sequences for simplicity or convert carefully if needed
        if elem.VR != "SQ":
            tag_name = elem.keyword if elem.keyword else str(elem.tag)
            dicom_dict[tag_name] = str(elem.value)
    # Pretty print JSON string with indentation
    return json.dumps(dicom_dict, indent=4)

def zoom_image(image, zoom_factor):
    """
    Zoom in/out by cropping and resizing the image.
    zoom_factor >1 => zoom in, <1 => zoom out
    """
    h, w = image.shape
    # Calculate crop size
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    # Crop center
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    cropped = image[top:top+new_h, left:left+new_w]
    # Resize back to original size
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return zoomed

def load_multi_dicom(uploaded_files):
    """
    Load multiple DICOM files, sort by InstanceNumber or SliceLocation, and stack images.
    Returns list of images and list of dicoms sorted.
    """
    dicoms = []
    for uploaded_file in uploaded_files:
        # Save temp file
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

    # Sort by InstanceNumber or SliceLocation
    def sort_key(x):
        dicom = x[0]
        return getattr(dicom, "InstanceNumber", 0) or getattr(dicom, "SliceLocation", 0) or 0
    dicoms.sort(key=sort_key)

    # Load images into list
    images = []
    for dicom, path in dicoms:
        img = dicom.pixel_array.astype(np.float32)
        img -= np.min(img)
        if np.max(img) != 0:
            img /= np.max(img)
        img *= 255.0
        images.append(img.astype(np.uint8))
    return images, dicoms

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

        # Display full metadata inside collapsible tab
        st.subheader("DICOM Metadata")
        with st.expander("Show full DICOM Metadata"):
            metadata_str = dicom_to_dict_str(current_dicom)
            st.code(metadata_str, language="json")

        st.subheader("Brightness / Contrast / Filter Controls")
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

        # Process image
        processed = apply_brightness_contrast(original_image, brightness, contrast)
        processed = apply_filter(processed, filter_type)
        processed = zoom_image(processed, zoom_factor)

        # Canny returns single channel, convert to RGB for display
        if filter_type == "Edge Detection":
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

        st.image(processed, caption=f"Slice {slice_idx+1} | Filter: {filter_type} | Zoom: {zoom_factor}x", use_container_width=True, clamp=True)

        st.markdown("#### Controls")
        st.markdown("""
        - Adjust **brightness** and **contrast** with sliders  
        - Select **filters** from dropdown menu  
        - Use **zoom** slider to zoom in/out  
        - Use **slice slider** (if multiple files) to navigate slices  
        """)

        # Clean up temp files
        for _, path in dicoms:
            if os.path.exists(path):
                os.remove(path)
    else:
        st.info("Please upload one or more DICOM files (.dcm) to begin.")

if __name__ == "__main__":
    main()
