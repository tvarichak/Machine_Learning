# 🧠 AI Programming Using Python – Workshop Repository

> **Instructor:** Rasoul Ameri  
> **Workshop Title:** AI Programming Using Python  
> **Focus Areas:** Python for Machine Learning • Data Preprocessing • Model Development • Explainable AI (SHAP)

---

## 🎯 Overview

This repository contains the complete materials from the **AI Programming Using Python** workshop conducted at the  
**International Graduate School of Artificial Intelligence (YunTech)**.

The program introduces participants to the **fundamentals of AI programming** through a hands-on, application-oriented approach.  
It builds essential skills in **data analysis**, **machine learning model development**, and **model interpretability**, forming the foundation for a professional career in AI and Data Science.

### Learning Outcomes
- Configure and manage Python environments using **Anaconda**  
- Utilize **NumPy**, **Pandas**, and **Matplotlib** for numerical computation and data visualization  
- Perform **data cleaning and preprocessing**  
- Implement core **machine learning algorithms** for classification and regression using **scikit-learn**  
- Apply **Explainable AI (XAI)** techniques using **SHAP** to interpret model predictions  

---

## 🗺️ Repository Structure for Machine learning via Explainability

The materials follow a **progressive learning roadmap**, designed to guide learners from basic programming toward advanced model interpretability and deployment.

| Phase | Topic | Folder | Key Materials | Status |
|-------|--------|---------|----------------|---------|
| 🧩 1 | **Environment Setup** | [1_Anaconda](./1_Anaconda) | [Anaconda](./1_Anaconda/11_Anaconda.ipynb) | ✅ |
| 🐍 2 | **Python Foundations** | [2_Python Tutorial](./2_Python%20Tutorial) | [Python Basics](./2_Python%20Tutorial/21_Python%20Basics.ipynb), [Numpy](./2_Python%20Tutorial/22_Numpy.ipynb), [Pandas](./2_Python%20Tutorial/23_Pandas.ipynb), [MatPlotlib](./2_Python%20Tutorial/24_MatPlotlib.ipynb) | ✅ |
| 🧹 3 | **Data Cleaning & Preparation** | [Data Cleaning and Preparation](./3_Data%20Cleaning%20and%20Preparation) | [Data Cleaning and Preparation](./3_Data%20Cleaning%20and%20Preparation/31_data_cleaning_preparation.ipynb) | ✅ |
| 🔍 4 | **Classification vs Regression** | [4_Classification Vs Regression](./4_Classification%20Vs%20Regression) | [Classification vs Regression [ppt]](./4_Classification%20Vs%20Regression/41_Classification.vs.Regression.pptx) | ✅ |
| 🤖 5 | **Supervised Learning Algorithms** | [5_Classification](./5_Classification) | Includes major classifiers such as Logistic Regression, KNN, SVM, Naive Bayes, Decision Tree, and Random Forest | ✅ |
| 🧠 6 | **Explainable AI (XAI)** | [SHAP](./5_Classification/52_KNN) | [SHAP](./5_Classification/52_KNN/5203_Shapey_values.ipynb) | ✅ |
| ⚙️ 7 | **Feature Engineering & Dimensionality Reduction** | *Coming Soon* | *(to be added)* | ⏳ |
| 🔧 8 | **Regression Algorithms** | *Coming Soon* | *(Linear, Polynomial, Ridge, Lasso)* | ⏳ |
| 🌐 9 | **Unsupervised Learning** | *Coming Soon* | *(K-Means, PCA, Hierarchical Clustering)* | ⏳ |
| 🚀 10 | **Deployment (MLOps)** | *Coming Soon* | *(Streamlit, Docker, CI/CD)* | ⏳ |
| 🔍 11 | **Advanced Explainable AI (LIME, DeepSHAP, ELI5)** | *Coming Soon* | *(to be added)* | ⏳ |

---

## 🤖 Module 5 – Classification Algorithms

This module covers the core supervised learning algorithms used in AI and Data Science projects.

| Algorithm | Folder | Key Notebooks |
|------------|---------|----------------|
| **Logistic Regression** | [51_Logistic Regression](./5_Classification/51_Logistic%20Regression) | [Logistic Regression](./5_Classification/51_Logistic%20Regression/5101_Logistic%20Regression.ipynb) |
| **K-Nearest Neighbors (KNN)** | [52_KNN](./5_Classification/52_KNN) | [1_KNN](./5_Classification/52_KNN/5201_KNN.ipynb), [2_KNN GridSearchCV](./5_Classification/52_KNN/5202_KNN%20GridSearchCV.ipynb), [3_Shapey_values](./5_Classification/52_KNN/5203_Shapey_values.ipynb) |
| **Support Vector Machine (SVM)** | [53 - SVM](./5_Classification/53_SVM) | [SVM](./5_Classification/53_SVM/5301_SVM.ipynb) |
| **Naive Bayes** | [54 - Naive Bayse](./5_Classification/54_Naive%20Bayse) | [Naive Bayse](./5_Classification/54_Naive%20Bayse/5401_Naive%20Bayse.ipynb) |
| **Decision Tree & Random Forest** | [55 - Decision Tree and Random Forest](./5_Classification/55_Decission%20Tree%20and%20Random%20Forest) | [Decission Tree and Random Forest](./5_Classification/55_Decission%20Tree%20and%20Random%20Forest/5501_Decission%20Tree%20and%20Random%20Forest.ipynb) |

---

## 🧩 Explainable AI (XAI)

**Explainable AI (XAI)** helps understand how models make decisions, improving transparency and trust.  
This workshop introduced **SHAP (SHapley Additive exPlanations)** to interpret model predictions at both the global and local level.

### Topics Covered
- Local and Global Interpretability  
- Feature Importance Visualization  
- SHAP Value Computation  
- Transparency in Non-Linear Models  
- Example Notebook → [3_Shapey_values.ipynb](./5_Classification/52_KNN/5203_Shapey_values.ipynb)

---
<!--
## 📚 Repository Structure

| Folder | Description | Key Files |
|---------|--------------|-----------|
| **1_Anaconda** | Environment setup and configuration | 11_Anaconda.ipynb |
| **2_Python Tutorial** | Python basics and core libraries | 21_Python Basics.ipynb, 22_Numpy.ipynb, 23_Pandas.ipynb, 24_MatPlotlib.ipynb |
| **3_Data Cleaning and Preparation** | Handling missing data, outliers, and preprocessing | 31_Data Cleaning and Preparation.ipynb |
| **4_Classification Vs Regression** | Conceptual overview comparing classifiers and regressors | Classification vs Regression.pptx |
| **5_Classification** | Practical implementation of supervised learning algorithms | Logistic Regression, KNN, SVM, Naive Bayes, Decision Tree, Random Forest |
| **6_Regression** | *(Coming soon)* Linear, Polynomial, Ridge, Lasso Regression | — |
| **7_Unsupervised Learning** | *(Coming soon)* K-Means, PCA, Hierarchical Clustering | — |
| **8_Deep Learning** | *(Coming soon)* MLP, CNN, RNN models | — |
| **9_Explainable AI (Advanced)** | *(Coming soon)* LIME, DeepSHAP, ELI5 | — |

---
-->
## 🔮 Future Additions

Planned topics to expand the **AI Programming and Machine Learning Engineer Roadmap** include:

- 📊 **Feature Engineering** & Dimensionality Reduction  
- 🔧 **Hyperparameter Optimization** (GridSearch, Bayesian Search)  
- 🧮 **Model Evaluation and Bias Detection**  
- ☁️ **MLOps and Streamlit Deployment**  
- 🔍 **Advanced Explainability Techniques (LIME, DeepSHAP, ELI5)**  

---

## 📫 Contact

**Rasoul Ameri**  
📧 [rasoulameri@gmail.com](mailto:rasoulameri90@gmail.com)  
🔗 [GitHub Profile](https://github.com/rasoulameri)

---

> 🧩 _This repository serves as a complete and evolving learning path for mastering **AI Programming Using Python**, guiding learners from environment setup to advanced explainable machine learning systems._
