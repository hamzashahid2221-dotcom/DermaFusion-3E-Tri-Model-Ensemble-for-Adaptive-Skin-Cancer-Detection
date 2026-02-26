# 🧴 AI-Assisted Detection of Skin Cancer Using Ensemble Deep Learning

This repository presents my research project on developing an **AI-assisted diagnostic model** for classifying skin cancer using **Deep Learning**.  
The system integrates **three independently trained CNN models** — **EfficientNetB0**, **ResNet50**, and **InceptionV3** — and combines their predictions through **model ensembling** to achieve higher diagnostic reliability.

---

## 🎯 Project Overview

Skin cancer diagnosis through dermoscopic images is challenging due to:
- subtle inter-class variations,
- high intra-class variability,
- and limited expert availability in low-resource clinical environments.

To tackle this challenge, I developed a **robust ensemble-based skin cancer classification system** capable of identifying:

- **Basal Cell Carcinoma (BCC)**
- **Melanoma**
- **Squamous Cell Carcinoma (SCC)**

By training **three separate deep neural networks** and averaging their prediction logits, the system reduces model bias, improves generalization, and enhances clinical reliability.

This work lies in the domain of **Medical Image Analysis (MIA)** and **AI-assisted dermatology**.

---

## 🧩 Dataset

The dataset consists of three clinically significant skin cancer classes:

| Class | Description |
|-------|-------------|
| Basal Cell Carcinoma | Common but low-risk skin cancer |
| Melanoma | Highly aggressive and life-threatening |
| Squamous Cell Carcinoma | Fast-growing, may metastasize |

**Dataset Split:**
- 80% Training  
- 20% Testing (used for ensemble evaluation)

Images were preprocessed according to each model’s input shape:
- **224×224** for EfficientNetB0  
- **224×224** for ResNet50  
- **299×299** for InceptionV3  

---

## 🧠 Model Architecture

Three deep learning architectures were trained **independently**:

### **1️⃣ EfficientNetB0**
- Lightweight and efficient  
- Pretrained on ImageNet  
- Excellent feature extraction for medical images  

### **2️⃣ ResNet50**
- Residual learning improves gradient flow  
- Strong baseline for image classification tasks  

### **3️⃣ InceptionV3**
- Factorized convolutions  
- Handles multi-scale features effectively  

Each model was fine-tuned on the skin dataset individually.

---

## 🤝 Model Ensembling

To boost diagnostic stability, the predictions from all three models were combined:

\[
P_{\text{final}} = \frac{P_{\text{EfficientNet}} + P_{\text{ResNet}} + P_{\text{Inception}}}{3}
\]

This **soft voting ensemble** reduces overfitting, increases robustness, and leverages strengths of each architecture.

---

## 🚀 Training Strategy

- **Independent training** of all three models  
- **ImageNet-pretrained weights**  
- **Fine-tuning** for medical feature extraction  
- **Batching + Prefetching** using TensorFlow Datasets  
- **Evaluation** performed on combined ensemble predictions  

---

## 📊 Results (Ensemble Model)

The ensemble model achieved highly reliable performance:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Basal Cell Carcinoma | 0.96 | 0.93 | 0.95 | 75 |
| Melanoma | 0.97 | 1.00 | 0.98 | 88 |
| Squamous Cell Carcinoma | 0.86 | 0.83 | 0.85 | 36 |
| **Overall Accuracy** | **0.94** | — | — | **199** |

### **Performance Summary**
- **Accuracy:** 94%  
- **Macro F1:** 0.92  
- **Weighted F1:** 0.94  

The model demonstrates **high precision** in melanoma detection and robust performance across all classes.

---

## 🔬 Interpretability

The model supports post-hoc explainability such as:
- **Grad-CAM**  
- **Grad-CAM++**  
- **Integrated Gradients**

These methods can highlight the lesion regions that contribute to the final prediction.

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|----------|
| TensorFlow / Keras | Deep Learning Models |
| NumPy, Pandas | Data Processing |
| Matplotlib, Seaborn | Visualization |
| scikit-learn | Evaluation & Metrics |
| Google Colab | Training & Experimentation |

---

## 📁 Repository Structure

---

## 🎓 Research Significance

This study highlights how **model ensembling** significantly improves performance in medical imaging tasks.  
It demonstrates the importance of:
- multi-architecture feature extraction,
- reducing false positives/negatives,
- and enhancing clinical decision support systems.

The approach aligns strongly with global trends in **AI for Dermatology** and **Medical Image Analysis**.

---

## 👨‍💻 Author

**Hamza Shahid**  
Bachelor of Biomedical Engineering
University of Engineering & Technology (UET), Lahore  

🔍 Research Interests:  
AI in Healthcare • Skin Cancer Detection • Ensemble Learning • Medical Imaging

---

## 📜 License

Released under the **MIT License** — free to use for research and academic purposes.

---

## 🌍 Acknowledgment

Special thanks to ISIC for datasets and the research community for enabling reproducible AI research.


(To be updated depending on your folder organization)

