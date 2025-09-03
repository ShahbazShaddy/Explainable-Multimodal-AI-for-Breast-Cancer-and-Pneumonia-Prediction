# ============================================
# Multimodal Late Fusion Inference + Explainability (SHAP + Grad-CAM)
# ============================================

import joblib
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import shap
import matplotlib.pyplot as plt
import numpy as np
import cv2

# ===========================
# 1. Load Models & Preprocessor
# ===========================

preprocessor = joblib.load("models/preprocessor.pkl")
cancer_model = joblib.load("models/cancer_model.pkl")
print("✅ Cancer model & preprocessor loaded successfully!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pneumonia_model = models.densenet121(weights=None)
num_features = pneumonia_model.classifier.in_features
pneumonia_model.classifier = nn.Linear(num_features, 1)
pneumonia_model.load_state_dict(
    torch.load("models/pneumonia_model.pth", map_location=device)
)
pneumonia_model = pneumonia_model.to(device)
pneumonia_model.eval()
print("✅ Pneumonia model loaded successfully!")

# ===========================
# 2. Preprocessing for Pneumonia Image
# ===========================

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(image_input):
    """Process either a file path or a file-like object"""
    if isinstance(image_input, str):
        # It's a file path
        img = Image.open(image_input).convert("RGB")
    else:
        # It's a file-like object (e.g., from Streamlit uploader)
        img = Image.open(image_input).convert("RGB")
    
    img_tensor = image_transforms(img).unsqueeze(0)
    return img_tensor.to(device), img

# ===========================
# 3. Prediction Functions
# ===========================

def predict_cancer(example_input: dict):
    df = pd.DataFrame([example_input])
    X_processed = preprocessor.transform(df)
    prob = cancer_model.predict_proba(X_processed)[0][1]
    return prob, df, X_processed

def predict_pneumonia(image_input):
    """Predict pneumonia from either a file path or a file-like object"""
    img_tensor, raw_img = preprocess_image(image_input)
    with torch.no_grad():
        logits = pneumonia_model(img_tensor)
        prob = torch.sigmoid(logits).item()
    return prob, img_tensor, raw_img

# ===========================
# 4. Fusion Strategy
# ===========================

def late_fusion(cancer_prob, pneumonia_prob, w_cancer=0.5, w_pneumonia=0.5):
    fused_prob = w_cancer * cancer_prob + w_pneumonia * pneumonia_prob
    return fused_prob

# ===========================
# 5. Extract Feature Names from Preprocessor
# ===========================

def get_feature_names(preprocessor):
    """Extract feature names after preprocessing - compatible with sklearn 1.7.1"""
    try:
        # Try direct method first (scikit-learn 1.0+)
        return preprocessor.get_feature_names_out()
    except (AttributeError, ValueError):
        # Fall back to manual extraction
        output_features = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'remainder':
                output_features.extend(cols)
            else:
                try:
                    # For transformers with get_feature_names_out method
                    if hasattr(transformer, 'get_feature_names_out'):
                        output_features.extend(transformer.get_feature_names_out(cols))
                    # Fall back to older method
                    elif hasattr(transformer, 'get_feature_names'):
                        output_features.extend(transformer.get_feature_names())
                    else:
                        # Just use column names as fallback
                        output_features.extend([f"{name}_{c}" for c in cols])
                except Exception as e:
                    print(f"Warning: Could not get feature names for {name}: {e}")
                    # Use generic feature names as fallback
                    output_features.extend([f"{name}_{i}" for i, _ in enumerate(cols)])
        return output_features

# ===========================
# 6. Explainability - SHAP for Cancer
# ===========================

def explain_cancer(model, X_processed, preprocessor):
    try:
        feature_names = get_feature_names(preprocessor)
    except Exception as e:
        print(f"Warning: Could not extract feature names: {e}")
        # Fallback to generic feature names
        feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
    
    # Use LinearExplainer for LogisticRegression
    if "LogisticRegression" in str(type(model)):
        explainer = shap.LinearExplainer(model, X_processed)
    else:
        explainer = shap.Explainer(model, X_processed)

    shap_values = explainer(X_processed)
    
    # Handle different SHAP values formats
    if hasattr(shap_values, 'values'):
        shap_array = shap_values.values[0]
    else:
        shap_array = shap_values[0]
    
    # Aggregate SHAP values back to original features
    grouped_importance = {}
    for i, val in enumerate(shap_array):
        if i < len(feature_names):
            # Extract base feature name
            parts = feature_names[i].split("_")
            base = parts[0]
            grouped_importance[base] = grouped_importance.get(base, 0) + abs(val)
        else:
            # Fallback
            grouped_importance[f"feature_{i}"] = abs(val)

    return grouped_importance

# ===========================
# 7. Explainability - Grad-CAM for Pneumonia
# ===========================

def grad_cam(model, img_tensor, raw_img, target_layer="features.denseblock4"):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    def forward_hook(module, input, output):
        activations.append(output.detach())

    # Register hooks
    layer = dict([*model.named_modules()])[target_layer]
    layer.register_forward_hook(forward_hook)
    layer.register_backward_hook(backward_hook)

    # Forward + backward
    output = model(img_tensor)
    loss = output[0][0]
    model.zero_grad()
    loss.backward()

    grads = gradients[0].cpu().numpy()[0]
    acts = activations[0].cpu().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for w, a in zip(weights, acts):
        cam += w * a
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, raw_img.size)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(raw_img), 0.6, heatmap, 0.4, 0)

    return overlay

# ===========================
# 8. Example Usage
# ===========================

example_input = {
    "Year": 2019,
    "Age": 42,
    "Menopause": 1,
    "Tumor Size (cm)": 3,
    "Inv-Nodes": 1,
    "Breast": "Left",
    "Metastasis": 0,
    "Breast Quadrant": "Upper outer",
    "History": 1
}

# Cancer prediction + SHAP
cancer_prob, df, X_processed = predict_cancer(example_input)
print(f"Cancer Probability: {cancer_prob:.4f}")
# grouped_importance = explain_cancer(cancer_model, X_processed, preprocessor)

# Pneumonia prediction + Grad-CAM
image_path = "scr/person1000_bacteria_2931.jpeg"
pneumonia_prob, img_tensor, raw_img = predict_pneumonia(image_path)
print(f"Pneumonia Probability: {pneumonia_prob:.4f}")
overlay = grad_cam(pneumonia_model, img_tensor, raw_img)

# Late fusion
final_prob = late_fusion(cancer_prob, pneumonia_prob)
print(f"Final Fused Probability: {final_prob:.4f}")

# ===========================
# 9. Display Explanations Side by Side
# ===========================

plt.figure(figsize=(6, 6))
plt.imshow(overlay)
plt.axis("off")
plt.title("Pneumonia (Image) - Grad-CAM")
plt.show()

# ===========================
# 10. Groq AI Integration for Human-Friendly Explanation
# ===========================

def get_groq_explanation(cancer_prob, pneumonia_prob, final_prob, example_input):
    """
    Get a human-friendly explanation of the prediction results from Groq AI.
    
    Args:
        cancer_prob: Probability of cancer
        pneumonia_prob: Probability of pneumonia
        final_prob: Combined risk probability
        example_input: Dictionary with patient information
        
    Returns:
        str: Human-friendly explanation from Groq AI
    """
    try:
        from groq import Groq
        import os
        from dotenv import load_dotenv
        
        # Load environment variables from .env file (same as reference app)
        load_dotenv()
        
        # Get API key using the same method and key name as reference app
        api_key = os.getenv("GROQ_API")
        
        # Debug message
        print(f"API key found: {'Yes' if api_key else 'No'}")
        
        # Generate default explanation in case API call fails
        default_explanation = f"""
        **Results Summary:**
        
        Based on the analysis, the cancer probability is {cancer_prob:.2%}, and the pneumonia probability is {pneumonia_prob:.2%}, resulting in a combined risk assessment of {final_prob:.2%}.
        
        **What This Means:**
        
        {'These results suggest a higher risk level that warrants prompt medical attention. Please consult with a healthcare professional to discuss these findings and appropriate next steps.' if final_prob > 0.5 else 'These results suggest a lower risk level, but regular check-ups are still recommended as part of ongoing health maintenance.'}
        
        **Important Note:**
        
        This is an automated assessment and should not replace professional medical advice. The accuracy of these predictions depends on many factors, including the quality of input data and limitations of the AI models used.
        """
        
        if not api_key:
            print("No API key found in environment variables")
            return default_explanation
        
        # Set up Groq client (same approach as working code)
        client = Groq(api_key=api_key)
        
        # Prepare the prompt with prediction results and patient data
        prompt = f"""
        I need a clear and compassionate explanation of medical prediction results for a patient in a consice way.
        
        Patient Information:
        - Age: {example_input.get('Age')} years old
        - Menopause status: {"Yes" if example_input.get('Menopause') == 1 else "No"}
        - Tumor Size: {example_input.get('Tumor Size (cm)')} cm
        - Invasive Nodes: {"Present" if example_input.get('Inv-Nodes') == 1 else "Not present"}
        - Affected Breast: {example_input.get('Breast')}
        - Metastasis: {"Yes" if example_input.get('Metastasis') == 1 else "No"}
        - Tumor Location: {example_input.get('Breast Quadrant')}
        - History of cancer: {"Yes" if example_input.get('History') == 1 else "No"}
        
        Prediction Results:
        - Cancer Probability: {cancer_prob:.2%}
        - Pneumonia Probability: {pneumonia_prob:.2%}
        - Combined Risk Assessment: {final_prob:.2%}
        
        Please provide:
        1. A simple, human-friendly explanation of what these results mean
        2. What the patient should consider doing next
        3. Important context or limitations they should know about these predictions
        
        Keep your response compassionate, clear, and about 3-5 paragraphs.
        """
        
        try:
            # Call Groq API
            print("Calling Groq API...")
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.3-70b-versatile",
            )
            
            # Return the explanation
            return chat_completion.choices[0].message.content
        except Exception as api_error:
            print(f"API call failed: {str(api_error)}")
            return f"⚠️ Error connecting to Groq API. Using default explanation instead.\n\n{default_explanation}"
        
    except Exception as e:
        print(f"Exception in get_groq_explanation: {str(e)}")
        return f"Unable to generate AI explanation. Please consult with a healthcare professional to interpret these results."