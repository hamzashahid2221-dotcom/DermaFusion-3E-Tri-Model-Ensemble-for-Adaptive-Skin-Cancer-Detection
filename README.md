# 🧴 AI-Assisted Skin Cancer Classification

This repository presents my research project on developing an **AI-assisted diagnostic model** for classifying **Skin Cancer types** using **Deep Learning**.  
The model leverages a **fine-tuned EfficientNetB0** architecture combined with an **Adaptive Categorical Focal Loss** to handle class imbalance and improve generalization.

---

## 🎯 Project Overview

Skin cancer is one of the most common forms of cancer globally.  
Early and accurate detection is crucial for **effective treatment** and **improved patient outcomes**.  
Manual examination and biopsy are **time-consuming** and often **subject to human error**, especially in regions with limited dermatological expertise.

To address this, I developed a **computer-aided diagnostic (CAD)** system capable of automatically classifying skin lesion images into:

- **Basal Cell Carcinoma**  
- **Melanoma**  
- **Squamous Cell Carcinoma**

This work contributes to the intersection of **AI and dermatology**, enabling faster and more reliable skin cancer diagnostics.

---

## 🧩 Dataset

The project uses the **ISIC dataset**, a publicly available skin lesion dataset curated for skin cancer classification.

**Dataset Description:**

| Source | Description | Classes |
|--------|-------------|--------|
| [ISIC Skin Cancer Dataset](https://www.isic-archive.com/) | Dermoscopic images of skin lesions | Basal Cell Carcinoma, Melanoma, Squamous Cell Carcinoma |

**Data Split:**
- 70% Training  
- 30% Validation  

**Classes:**
- Basal Cell Carcinoma  
- Melanoma  
- Squamous Cell Carcinoma  

**Evaluation Metrics:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Basal Cell Carcinoma | 0.89 | 0.91 | 0.90 | 75 |
| Melanoma | 0.97 | 0.99 | 0.98 | 88 |
| Squamous Cell Carcinoma | 0.88 | 0.81 | 0.84 | 36 |
| **Overall Accuracy** | **0.92** | **0.92** | **0.92** | 199 |
| **Macro Avg** | 0.91 | 0.90 | 0.91 | 199 |
| **Weighted Avg** | 0.92 | 0.92 | 0.92 | 199 |

---

## 🧠 Model Architecture

**Base Model:** EfficientNetB0 (pretrained on ImageNet)

- The base model is **initially frozen** to leverage pretrained features.  
- The **last 20 layers are later unfrozen** for **fine-tuning**, enabling learning of dermatological image patterns.  
- Input images are preprocessed and resized to match EfficientNetB0 requirements.

---

## ⚙️ Adaptive Categorical Focal Loss

To handle **class imbalance**, particularly due to fewer squamous cell carcinoma samples, the model uses **adaptive categorical focal loss**:

\[
FL(p_t) = - \alpha_t (1 - p_t)^{\gamma_t} \log(p_t)
\]

Where:  
- \( p_t \): Predicted probability of the true class  
- \( \alpha_t \): Adaptive class weight (updates per epoch)  
- \( \gamma_t \): Focusing parameter emphasizing hard-to-classify examples  

This ensures balanced performance across all classes.
Key Innovations:
1. **Dynamic per-class adaptation:**  
   - Unlike standard focal loss, the **class weight** (\(\alpha_t\)) and **focusing parameter** (\(\gamma_t\)) are **adjusted during training** based on class-wise performance metrics (e.g., recall).  
   - This allows the model to focus more on harder-to-classify or underrepresented classes.

2. **Self-adjusting loss mechanism:**  
   - The adaptive loss forms a **feedback loop**, where model performance informs how the loss function emphasizes each class.  
   - This reduces manual hyperparameter tuning and promotes balanced learning.

3. **Stable updates:**  
   - Adjustments are applied carefully to prevent abrupt changes, ensuring **robust and stable training**.

> **Note:** The full implementation details of this adaptive loss function are part of ongoing research and are available upon request or will be published in a forthcoming manuscript.
---

## 🚀 Training Strategy

- **Optimizer:** Adam (LR = 0.001 → adaptive reduction)  
- **Batch Size:** 32  
- **EarlyStopping:** Patience = 4 (restore best weights)  
- **ModelCheckpoint:** Saves best model on validation loss  
- **Fine-Tuning Phase:** Unfreeze last 20 EfficientNetB0 layers  

---

## 📊 Results

The model achieves high performance for skin cancer classification:

- **Macro Avg F1:** 0.91  
- **Weighted Avg F1:** 0.92  
- **Validation Accuracy:** 92%  

The system demonstrates **robust performance** despite class imbalance and limited dataset size.

---

## 🔬 Interpretability

Post-hoc interpretability can be added using:

- **Grad-CAM / Grad-CAM++** to visualize discriminative regions in dermoscopic images  
- **Integrated Gradients** for pixel-level contribution analysis  

These methods enhance **clinical trust** and **explainability** of the AI model.

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| TensorFlow / Keras | Deep Learning Framework |
| NumPy, Pandas | Data Handling |
| Matplotlib, Seaborn | Visualization |
| scikit-learn | Evaluation Metrics |
| ISIC Dataset | Dermoscopic Images for Skin Cancer |

---

## 📁 Repository Structure

---

## 🎓 Research Significance

This project demonstrates how **transfer learning with adaptive focal loss** can effectively classify **skin cancer types**.  
The model provides a **scalable diagnostic tool**, aiding dermatologists in early detection and improving patient outcomes.

---

## 👨‍💻 Author

**Hamza Shahid**  
Bachelor of Biomedical Engineering (with Distinction)  
University of Engineering & Technology (UET), Lahore  

🔍 Research Interests:  
AI in Healthcare • Medical Image Analysis • Deep Learning for Diagnostics  

---

## 📜 License

This repository is released under the **MIT License** — freely available for academic and research purposes.

---

## 🌍 Acknowledgment

Special thanks to the **ISIC dataset contributors** and the open-source AI community for enabling reproducible skin cancer research.



