import streamlit as st
from kraken import blla, rpred
from kraken.lib import vgsl
from kraken.lib import models
from PIL import Image, ImageDraw

# Define available OCR models for segmentation and recognition
segmentation_models = {
    "1col_442_sam_v1.mlmodel": "models/1col_442_sam_v1.mlmodel",
    "ubma_sam_v4.mlmodel": "models/ubma_sam_v4.mlmodel"
}

recognition_models = {
    "sinai_sam_rec_v4.mlmodel": "models/sinai_sam_rec_v4.mlmodel",
    "sinai_sam_rec_v2.mlmodel": "models/sinai_sam_rec_v2.mlmodel"
}

# Streamlit app title and description
st.title("OCR with Kraken - Segmentation and Recognition")
st.write("Upload an image, select segmentation and recognition models, and view OCR results.")

# Upload image file
uploaded_image = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])

# Select segmentation and recognition models
selected_seg_model = st.selectbox("Select Kraken Segmentation Model", list(segmentation_models.keys()))
selected_rec_model = st.selectbox("Select Kraken Recognition Model", list(recognition_models.keys()))

# Option to draw baselines
draw_baselines = st.radio("Options", ("Do not draw baselines", "Draw baselines")) == "Draw baselines"

# Process the image if uploaded and models selected
if uploaded_image and selected_seg_model and selected_rec_model:
    # Load the image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Load selected Kraken segmentation and recognition models
    seg_model_path = segmentation_models[selected_seg_model]
    rec_model_path = recognition_models[selected_rec_model]
    seg_model = vgsl.TorchVGSLModel.load_model(seg_model_path)
    rec_model = models.load_any(rec_model_path)

    # Segment image using Kraken segmentation model
    baseline_seg = blla.segment(image, model=seg_model)

    # Pass segmentation result to recognition model
    pred_it = rpred.rpred(network=rec_model, im=image, bounds=baseline_seg)

    # Prepare to draw boundaries and display info
    boundaries_info = []
    draw = ImageDraw.Draw(image)

    # Process recognition predictions for lines and draw on image
    for idx, pred in enumerate(pred_it):
        prediction = pred.prediction
        line_boundary = [(int(x), int(y)) for x, y in pred.boundary]
        line_baseline = [(int(x), int(y)) for x, y in pred.baseline] if pred.baseline else None
        line_type = pred.tags.get("type", "undefined")  # Get line type dynamically if available

        # Add boundary, baseline (if selected), and prediction to display info in the new order
        boundaries_info.append(f"**Line {idx + 1}** (type: {line_type}):\n  - Boundary: {line_boundary}")

        # Draw boundary in green
        draw.polygon(line_boundary, outline="green")

        # Draw baseline if the option is selected and add it to display info
        if draw_baselines and line_baseline:
            boundaries_info.append(f"  - Baseline: {line_baseline}")
            draw.line(line_baseline, fill="red", width=2)  # Draw baseline in red

        # Add prediction last
        boundaries_info.append(f"  - Prediction: {prediction}")

    # Process and draw region boundaries from baseline_seg
    for region_type, region_list in baseline_seg.regions.items():
        for idx, region_data in enumerate(region_list):
            if hasattr(region_data, "boundary"):
                region_boundary = [(int(x), int(y)) for x, y in region_data.boundary]
                region_type_name = region_data.tags.get("type", region_type)  # Get region type dynamically
                boundaries_info.append(f"**Region {idx + 1}** (type: {region_type_name}):\n  - Boundary: {region_boundary}")
                draw.polygon(region_boundary, outline="blue")  # Draw region boundary in blue

    # Display the image with boundaries drawn
    st.image(image, caption="Image with OCR boundaries (green for lines, blue for regions), baselines (red if selected)", use_column_width=True)

    # Display the list of boundaries, predictions, and baselines
    st.write("**List of Boundaries, Predictions, and Baselines (if selected):**")
    for info in boundaries_info:
        st.write(info)

