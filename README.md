# 🧠 TumorSense AI

*Advanced Brain Tumor Detection using Deep Learning & Explainable AI*

---

# Live Application

🔗 **Try it here:** [https://brain-tumor-ai.streamlit.app/](https://brain-tumor-ai.streamlit.app/)

TumorSense AI is an AI-powered medical imaging system that detects and classifies brain tumors from MRI scans using deep learning and explainable AI techniques.

Built for academic research, demonstration, and intelligent diagnostic assistance.

---

✨ Key Features

🧠 **MRI Brain Tumor Classification**
Detects 4 classes:

* Glioma
* Meningioma
* Pituitary
* No Tumor

📊 **Confidence Score & Probability Matrix**
Displays prediction confidence and full probability breakdown for all classes.

🔥 **Grad-CAM Explainability (Medical Visualization)**
Highlights the exact region where AI focuses inside MRI for transparent decision-making.

🎨 **Advanced Clinical UI**
Modern, high-tech diagnostic interface built with Streamlit + custom CSS.

☁️ **Cloud Model Hosting (HuggingFace)**
Model hosted externally for lightweight GitHub deployment.

---

🏗 Tech Stack

**Frontend & Deployment**

* Streamlit (Web App UI)
* Custom CSS (Cyber-clinical interface)
* Streamlit Cloud hosting

**AI & Deep Learning**

* TensorFlow / Keras
* ResNet50 Transfer Learning
* Grad-CAM Explainable AI
* NumPy & OpenCV

**Cloud & Storage**

* HuggingFace (Model hosting)
* GitHub (Code repository)

---

## 🧪 Model Performance

| Metric       | Score       |
| ------------ | ----------- |
| Accuracy     | **~88%**    |
| Classes      | 4           |
| Architecture | ResNet50    |
| Input Size   | 224×224 MRI |

Includes explainability heatmaps for medical interpretability.

---

## 📁 Project Structure

```
TumorSense-AI/
│
├── app.py                # Streamlit application
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
└── model (hosted on HuggingFace)
```

---

## ⚙️ Installation (Run Locally)

### 1️⃣ Clone repository

```bash
git clone https://github.com/your-username/TumorSense-AI.git
cd TumorSense-AI
```

### 2️⃣ Install requirements

```bash
pip install -r requirements.txt
```

### 3️⃣ Run app

```bash
streamlit run app.py
```

---

## 🌐 Model Hosting

Model stored securely on HuggingFace:

```
brain_tumor_final.keras
```

Loaded dynamically inside app to bypass GitHub size limits.

---

## ⚠️ Disclaimer

This system is developed for:

* Academic research
* AI demonstration
* Educational purposes

**Not intended for real clinical diagnosis.**
Always consult certified medical professionals.

---

## 👨‍💻 Author

SUDHA S
AI Developer | Full Stack Learner | Deep Learning Enthusiast

Built as final-year AI project integrating:

* Deep Learning
* Explainable AI
* Cloud Deployment
* Modern UI/UX

---

## ⭐ Future Enhancements

* Doctor dashboard
* Patient report PDF export
* Multi-scan upload
* 3D MRI support
* Hospital integration API

---

## 💡 If you like this project

Star ⭐ the repo and share feedback!

**TumorSense AI — Intelligent MRI Diagnostics Powered by Deep Learning**
