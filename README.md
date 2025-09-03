<img alt="Project Workflow" src="scr\img.png">

# Explainable Multimodal AI for Breast Cancer and Pneumonia Prediction

A multimodal deep learning approach for medical diagnosis that combines clinical data and medical imaging with explainable AI techniques.

##  Overview

This project leverages both structured clinical data and medical imaging to create a more robust diagnostic system for two critical conditions:

- Breast Cancer Diagnosis (using clinical data)
- Pneumonia Detection (using chest X-ray images)

The system combines predictions from both modalities through late fusion, providing not only accurate predictions but also human-interpretable explanations using `LLM` and technique like `Grad-CAM`.

## Repository Structure

```
├── build_unimodal_models/
│   ├── breast-cancer-diagnosis-prediction.ipynb
│   └── chest-x-ray-pneumonia-classification.ipynb
├── models/
│   ├── cancer_model.pkl
│   ├── pneumonia_model.pth
│   └── preprocessor.pkl
├── scr/
│   └── person1000_bacteria_2931.jpeg (example image)
├── .env
├── multimodal_fusion.py
├── streamlit_app.py
└── README.md
```

# Datasets

## Breast Cancer Dataset
- **Source:** Clinical data with demographic and tumor characteristics (https://www.kaggle.com/datasets/fatemehmehrparvar/breast-cancer-prediction)
- **Features:** Age, menopause status, tumor size, invasive nodes, breast quadrant, etc.
- **Target:** Binary classification (Benign vs. Malignant)
- **Preprocessing:**
  - Missing value imputation
  - Normalization of numeric features
  - One-hot encoding of categorical features
  - Train/Validation/Test split (64%/16%/20%)

## Chest X-ray Dataset
- **Source:** Chest X-ray Pneumonia Dataset (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes:** NORMAL (1,341 images) and PNEUMONIA (3,875 images)
- **Structure:** Train, Validation, and Test subsets
- **Preprocessing:**
  - Resizing to 224×224 pixels
  - Normalization using ImageNet mean/std
  - Data augmentation (random crops, rotations, brightness adjustments)
  - Class imbalance handling with WeightedRandomSampler and weighted loss

# Methodology

1. **Unimodal Models**
   - **Cancer Diagnosis Model**
     - **Models Evaluated:** Logistic Regression, Random Forest, SVM, XGBoost, KNN, Naive Bayes, Gradient Boosting
     - **Best Model:** Logistic Regression (97.06% validation accuracy)
     - **Metrics:** Accuracy, AUROC, F1-score
   - **Pneumonia Detection Model**
     - **Architecture:** DenseNet121 with transfer learning
     - **Training:**
       - Fine-tuned on chest X-ray images
       - BCEWithLogitsLoss with class weights
       - AdamW optimizer with weight decay
       - OneCycleLR scheduler
     - **Performance:** 90.71% test accuracy, 97.16% AUROC

2. **Multimodal Fusion**
   - **Approach:** Late fusion of prediction probabilities
   - **Formula:** final_prob = w_cancer * cancer_prob + w_pneumonia * pneumonia_prob
   - **Weights:** Configurable (default: equal weighting)

3. **Explainability (XAI)**
   - **Image Data:** Grad-CAM to highlight regions of interest in X-rays
   - **Narrative Explanation:** AI-generated clinical interpretation using Groq LLM API

# Web Application

The project includes a Streamlit web application for interactive prediction:

- **Input:**
  - Upload chest X-ray image
  - Enter patient clinical data
  - Option to load example data
- **Output:**
  - Cancer probability
  - Pneumonia probability
  - Combined risk assessment
  - Grad-CAM visualization
  - AI-generated clinical explanation

# Setup and Installation

- Clone the repository.
    ```
    git clone https://github.com/ShahbazShaddy/Explainable-Multimodal-AI-for-Breast-Cancer-and-Pneumonia-Prediction.git
    cd Explainable-Multimodal-AI-for-Breast-Cancer-and-Pneumonia-Prediction
    ```
- Create and activate a virtual environment
    ```
    # Using conda
    conda create -n explainable-ai python=3.8
    conda activate explainable-ai

    # Or using venv
    python -m venv env
    # On Windows
    env\Scripts\activate
    # On Unix or MacOS
    source env/bin/activate
    ```
- Install dependencies
    ```
    pip install -r requirements.txt
    ```
- Set up API key for explanations

    Create a .env file with your Groq API key:
    ```
    GROQ_API=your_groq_api_key_here
    ```
- Run the Streamlit app
    ```
    streamlit run streamlit_app.py
    ```
# Usage
Access the web application at [http://localhost:8501](http://localhost:8501)  
Upload a chest X-ray image or use the example image  
Enter patient data or click "Load Example Data"  
Click "Predict" to generate the analysis  
Review the prediction results, visualizations, and explanations  

# Technologies Used
- **Python**: Core programming language
- **PyTorch**: Deep learning framework for the pneumonia model
- **scikit-learn**: Machine learning for the cancer model
- **Streamlit**: Web application framework
- **SHAP**: Explainability for tabular data
- **Grad-CAM**: Explainability for image data
- **Groq API**: Large language model for generating clinical explanations
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization

# Results
## Unimodal Performance
| Model                                         | Test Accuracy | Validation Accuracy |
|-----------------------------------------------|---------------|-------|
| Cancer Diagnosis (Logistic Regression)       | 90.70%       | 97.06% |
| Pneumonia Detection (DenseNet121)            | 90.71%       | 94.00% |

## Multimodal Performance
- **Combined Risk Assessment**: Improved predictive power over individual models  
- **Explainability**: Enhanced clinical utility through visual and textual explanations  

# Future Work
- Real-time inference for faster predictions  
- Additional modalities (e.g., patient history, genomic data)  
- More sophisticated fusion techniques (early fusion, feature-level fusion)  
- Deployment to cloud services for accessibility  
- Mobile application for point-of-care use  
- Expanded dataset for improved generalization  

# Disclaimer
> This tool is for educational and research purposes only and should not replace professional medical advice. Always consult healthcare professionals for diagnosis and treatment decisions.