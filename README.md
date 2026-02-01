<div align="center">

<!-- Animated Header -->
<img src="https://readme-typing-svg.herokuapp.com?font=Outfit&weight=700&size=40&pause=1000&color=6366F1&center=true&vCenter=true&width=600&lines=Telecom+Retention+System;Predict.+Analyze.+Retain.;AI-Powered+Customer+Intelligence" alt="Typing SVG" />

<br/>

### **The Enterprise-Grade Solution for Churn Prevention**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-white?style=for-the-badge&logo=flask&logoColor=black)](https://flask.palletsprojects.com)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.0%2B-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Vercel](https://img.shields.io/badge/Deployed_on-Vercel-black?style=for-the-badge&logo=vercel&logoColor=white)](https://vercel.com)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=for-the-badge)](https://github.com/karthik-vana/Telecom-Retention-System/graphs/commit-activity)

<br/>

[ **🔴 Live Platform** ](https://customer-retention-prediction-alpha.vercel.app/) &nbsp;&nbsp;|&nbsp;&nbsp; [ **� View Report** ](#model-performance) &nbsp;&nbsp;|&nbsp;&nbsp; [ **� Report Issue** ](issues)

<br/>

</div>

---

## 🧐 **The Problem**
In the subscription economy, **Customer Churn** (the rate at which customers leave) is the #1 silent revenue killer. Most telecom companies react *after* a customer cancels. 

## � **Our Solution**
The **Telecom Retention System** flips the script. It uses advanced Machine Learning to identify at-risk customers *months before they leave*.

> **"It's not just a prediction. It's an intervention."**

This system:
1.  **Ingests** customer data (billing, tenure, services).
2.  **Predicts** churn probability in real-time.
3.  **Explains** the root cause (e.g., "High Fiber Optic costs").
4.  **Recommends** a specific retention strategy (e.g., "Offer 15% discount").

<br/>

## ✨ **System Capabilities**

| Capability | What it does | Tech Involved |
| :--- | :--- | :--- |
| **🧠 Intelligent Inference** | Predicts customer behavior with **96% Accuracy**. | Random Forest, Scikit-Learn |
| **⚡ Real-Time API** | Processes requests in **< 100ms** via serverless functions. | Flask, Vercel Edge |
| **� Pattern Recognition** | Identifies non-linear relationships between 20+ variables. | Feature Engineering, NumPy |
| **⚖️ Smart Balancing** | Handles imbalanced data to ensure fair predictions. | SMOTE (Synthetic Minority Over-sampling) |
| **🛡️ Secure Privacy** | Processes data ephemerally without storing PII. | Stateless Architecture |

<br/>

## 🏗️ **System Architecture**

```mermaid
graph TD
    A[User/Client] -->|HTTP Request| B(Vercel Edge Network)
    B -->|Route /predict| C{Flask API}
    C -->|Preprocess| D[Data Cleaning & Encoding]
    D -->|Scale| E[RobustScaler]
    E -->|Inference| F[Random Forest Model]
    F -->|Probability & Factors| C
    C -->|JSON Response| A
```

</div>

<br/>

<details>
<summary><b>� Click to view Project Directory Structure</b></summary>
<br/>

```bash
Telecom-Retention-System/
├── api/
│   └── index.py          # Serverless Entry Point
├── app/
│   ├── static/           # CSS, JS, Images
│   └── templates/        # HTML5 Templates (Jinja2)
├── model/
│   ├── model.pkl         # Trained Random Forest Model
│   └── scaler.pkl        # Fitted RobustScaler
├── notebooks/            # Jupyter Notebooks for EDA
├── requirements.txt      # Python Dependencies
├── vercel.json           # Cloud Configuration
└── README.md             # Documentation
```
</details>

<br/>

## 🚀 **Quick Start Guide**

Run this system on your local machine in **3 simple steps**.

### 1. Clone & Enter
```bash
git clone https://github.com/karthik-vana/Telecom-Retention-System.git
cd Telecom-Retention-System
```

### 2. Install Engine
```bash
pip install -r requirements.txt
```

### 3. Ignite
```bash
python index.py
```
> The dashboard will launch at `http://localhost:5000`

<br/>

## 🧪 **Model Performance Metrics**

We validated the model using a 20% hold-out test set. 

-   **Accuracy**: `0.96` (Correctly classifies 96/100 customers)
-   **Precision**: `0.94` (Minimizes false alarms for expensive retention offers)
-   **Recall**: `0.93` (Captures the vast majority of churning customers)

<br/>

## 👨‍💻 **Creator & Mainteiner**

<div align="center">

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMjR4M2Z4NnZ4M3Z4M3Z4M3Z4M3Z4M3Z4M3Z4M3Z4M3Z4/qgQUggAC3Pfv687qPC/giphy.gif" width="30">

### **Karthik Vana**
**Data Engineer | ML Engineer | AI Engineer**

<a href="https://linkedin.com/in/karthik-vana"><img src="https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white"></a>
<a href="https://github.com/karthik-vana"><img src="https://img.shields.io/badge/GitHub-Follow-black?style=for-the-badge&logo=github&logoColor=white"></a>

</div>

<br/>

---

<div align="center">
    <i>Made with ❤️ and Python. © 2025 Telecom Retention System.</i>
</div>
