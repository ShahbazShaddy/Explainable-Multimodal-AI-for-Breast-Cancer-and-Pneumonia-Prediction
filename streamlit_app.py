import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import time
import os
from multimodal_fusion import (
    predict_cancer, 
    predict_pneumonia, 
    late_fusion, 
    explain_cancer, 
    grad_cam,
    preprocessor, 
    cancer_model,
    pneumonia_model,
    get_groq_explanation
)

# Set page configuration
st.set_page_config(
    page_title="Explainable Multimodal AI for Breast Cancer and Pneumonia Prediction",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state for form values
if 'year' not in st.session_state:
    st.session_state.year = 2019
if 'age' not in st.session_state:
    st.session_state.age = 45
if 'menopause' not in st.session_state:
    st.session_state.menopause = "No"
if 'tumor_size' not in st.session_state:
    st.session_state.tumor_size = 2.0
if 'inv_nodes' not in st.session_state:
    st.session_state.inv_nodes = "No"
if 'breast' not in st.session_state:
    st.session_state.breast = "Left"
if 'metastasis' not in st.session_state:
    st.session_state.metastasis = "No"
if 'breast_quadrant' not in st.session_state:
    st.session_state.breast_quadrant = "Upper outer"
if 'history' not in st.session_state:
    st.session_state.history = "No, I haven't"
if 'use_example_image' not in st.session_state:
    st.session_state.use_example_image = False

# Header
st.title("🏥 Explainable Multimodal AI for Breast Cancer and Pneumonia Prediction")
st.markdown("Upload an chest X-ray image and provide patient information for multimodal prediction")

# Create two columns
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Patient Information")
    
    # Add a "Load Example Data" button
    if st.button("Load Example Data"):
        st.session_state.year = 2019
        st.session_state.age = 42
        st.session_state.menopause = "Yes"
        st.session_state.tumor_size = 3.0
        st.session_state.inv_nodes = "Yes"
        st.session_state.breast = "Left"
        st.session_state.metastasis = "No"
        st.session_state.breast_quadrant = "Upper outer"
        st.session_state.history = "Yes, I have it previously"
        st.session_state.use_example_image = True
        st.success("Example data loaded! Submit the form to see results.")
    
    # Create a form for input data
    with st.form("patient_form"):
        # Year selection
        year = st.selectbox("Year", options=list(range(1999, 2076)), index=list(range(1999, 2076)).index(st.session_state.year))
        
        # Age input with validation
        age = st.number_input(
            "Age", 
            min_value=0, 
            max_value=150, 
            value=st.session_state.age,
            help="Patient's age (0-150)"
        )
        
        # Menopause
        menopause = st.selectbox(
            "Menopause", 
            options=["No", "Yes"], 
            index=0 if st.session_state.menopause == "No" else 1,
            help="Whether the patient has reached menopause"
        )
        menopause_val = 1 if menopause == "Yes" else 0
        
        # Tumor Size
        tumor_size = st.number_input(
            "Tumor Size (cm)", 
            min_value=0.0, 
            max_value=20.0, 
            value=st.session_state.tumor_size, 
            step=0.1,
            help="Size of tumor in centimeters"
        )
        
        # Inv-Nodes
        inv_nodes = st.selectbox(
            "Inv-Nodes", 
            options=["No", "Yes"],
            index=0 if st.session_state.inv_nodes == "No" else 1,
            help="Whether invasive nodes are present"
        )
        inv_nodes_val = 1 if inv_nodes == "Yes" else 0
        
        # Breast
        breast = st.selectbox(
            "Breast", 
            options=["Left", "Right"],
            index=0 if st.session_state.breast == "Left" else 1,
            help="Which breast is affected"
        )
        
        # Metastasis
        metastasis = st.selectbox(
            "Metastasis", 
            options=["No", "Yes"],
            index=0 if st.session_state.metastasis == "No" else 1,
            help="Whether cancer has spread to other parts of the body"
        )
        metastasis_val = 1 if metastasis == "Yes" else 0
        
        # Breast Quadrant
        breast_quadrant_options = ["Upper outer", "Lower outer", "Upper inner", "Lower inner", "#"]
        breast_quadrant = st.selectbox(
            "Breast Quadrant", 
            options=breast_quadrant_options,
            index=breast_quadrant_options.index(st.session_state.breast_quadrant),
            help="Location of the tumor in the breast"
        )
        
        # History
        history_options = ["No, I haven't", "Yes, I have it previously"]
        history = st.selectbox(
            "History", 
            options=history_options,
            index=history_options.index(st.session_state.history),
            help="Whether the patient has a history of cancer"
        )
        history_val = 1 if history == "Yes, I have it previously" else 0
        
        # X-ray image upload
        st.subheader("Upload X-ray Image")
        uploaded_file = st.file_uploader(
            "Choose an X-ray image file", 
            type=["jpg", "jpeg", "png"],
            help="Upload a chest X-ray image for pneumonia prediction"
        )
        
        # Submit button
        submit_button = st.form_submit_button("Predict")

# Display results in the second column
with col2:
    if submit_button:
        st.subheader("Analysis Results")
        
        # Create a progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        # Prepare input data
        example_input = {
            "Year": year,
            "Age": age,
            "Menopause": menopause_val,
            "Tumor Size (cm)": tumor_size,
            "Inv-Nodes": inv_nodes_val,
            "Breast": breast,
            "Metastasis": metastasis_val,
            "Breast Quadrant": breast_quadrant,
            "History": history_val
        }
        
        # Process X-ray image - either uploaded or example
        image_data = None
        if uploaded_file is not None:
            # Convert the file to an in-memory bytes object
            image_data = io.BytesIO(uploaded_file.getvalue())
        elif st.session_state.use_example_image:
            # Use the example image
            example_image_path = "scr/person1000_bacteria_2931.jpeg"
            if os.path.exists(example_image_path):
                with open(example_image_path, "rb") as f:
                    image_data = io.BytesIO(f.read())
            else:
                st.error(f"Example image not found at {example_image_path}")
        
        if image_data is not None:
            # Update progress
            progress_text.text("Processing X-ray image...")
            progress_bar.progress(25)
            
            # Pneumonia prediction
            pneumonia_prob, img_tensor, raw_img = predict_pneumonia(image_data)
            
            # Generate Grad-CAM visualization
            progress_text.text("Generating explanations...")
            progress_bar.progress(50)
            overlay = grad_cam(pneumonia_model, img_tensor, raw_img)
            
            # Cancer prediction
            progress_text.text("Processing patient data...")
            progress_bar.progress(75)
            cancer_prob, df, X_processed = predict_cancer(example_input)
            
            # Generate SHAP explanations
            try:
                importance = explain_cancer(cancer_model, X_processed, preprocessor)
                has_importance = True
            except Exception as e:
                st.error(f"Could not generate feature importance: {e}")
                has_importance = False
            
            # Late fusion
            final_prob = late_fusion(cancer_prob, pneumonia_prob)
            
            # Update progress
            progress_text.text("Completed!")
            progress_bar.progress(100)
            time.sleep(0.5)  # Short pause for UX
            progress_bar.empty()
            progress_text.empty()
            
            # Reset example image flag after use
            st.session_state.use_example_image = False
            
            # Display results
            results_col1, results_col2 = st.columns(2)
            
            with results_col1:
                st.subheader("Prediction Results")
                st.metric("Cancer Probability", f"{cancer_prob:.2%}")
                st.metric("Pneumonia Probability", f"{pneumonia_prob:.2%}")
                st.metric("Combined Risk (Fusion)", f"{final_prob:.2%}")
                
                # Provide interpretation
                st.subheader("Interpretation")
                if final_prob > 0.75:
                    st.error("High Risk: Immediate medical consultation recommended")
                elif final_prob > 0.5:
                    st.warning("Moderate Risk: Medical consultation recommended")
                else:
                    st.success("Lower Risk: Consider regular check-ups")
                
                # Add AI explanation with Groq
                st.subheader("AI-Generated Explanation")
                with st.spinner("Generating detailed explanation..."):
                    ai_explanation = get_groq_explanation(
                        cancer_prob, 
                        pneumonia_prob, 
                        final_prob, 
                        example_input
                    )
                st.markdown(ai_explanation)
                st.info("This explanation is generated by AI and should not replace professional medical advice.")
                
                # Display feature importance if available
                # if has_importance:
                #     st.subheader("Key Factors (Cancer)")
                #     importance_df = pd.DataFrame({
                #         'Feature': list(importance.keys()),
                #         'Importance': list(importance.values())
                #     }).sort_values('Importance', ascending=False)
                    
                #     # Create and display the bar chart
                #     fig, ax = plt.subplots(figsize=(10, 5))
                #     ax.barh(importance_df['Feature'], importance_df['Importance'])
                #     ax.set_xlabel('Feature Importance')
                #     ax.set_title('SHAP Feature Importance')
                #     st.pyplot(fig)
            
            with results_col2:
                st.subheader("X-ray Analysis")
                
                # Create tabs for raw image and overlay with Grad-CAM as default
                tab1, tab2 = st.tabs(["Activation Heatmap (Grad-CAM)", "Original X-ray"])
                
                with tab1:
                    # Convert the overlay to RGB format for display
                    st.image(overlay, caption="Areas of interest highlighted", use_container_width=True)
                    st.info("Highlighted areas show regions the model focused on for pneumonia detection")
                
                with tab2:
                    st.image(raw_img, caption="Original X-ray", use_container_width=True)
        else:
            st.error("Please upload an X-ray image or use the example data to proceed with prediction")

# Add information and disclaimer at the bottom
st.markdown("---")
st.caption("""
**Disclaimer**: This tool is for educational purposes only and should not replace professional medical advice. 
Always consult healthcare professionals for diagnosis and treatment decisions.
""")
