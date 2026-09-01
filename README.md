# 🇮🇳 Indian Name Gender Classifier

A character-level deep learning system that predicts the likely gender associated with an Indian name using a **Bidirectional LSTM (Bi-LSTM) ensemble**.

The project covers the complete machine-learning workflow:

**Data Collection → Data Cleaning → Exploratory Analysis → Model Training → Model Comparison → Evaluation → Ensemble → Web Application → Deployment**

---

## 🚀 Live Demo

**Live Application:**  
https://indiangenderclassifier.onrender.com

---

## 📌 Project Overview

Names often contain useful linguistic and character-level patterns that can be used for classification.

This project uses a **character-level neural network** to learn these patterns from Indian names and classify a name as:

- 👨 Male
- 👩 Female

Instead of relying on manually created rules such as name endings, the model learns character-level patterns directly from the training data.

The final web application allows a user to enter an Indian name and receive:

- Predicted gender
- Male probability
- Female probability
- Prediction confidence
- Confidence level
- Individual model probabilities

---

# 🧠 Machine Learning Approach

The primary model used in the project is a **Bidirectional LSTM (Bi-LSTM)**.

### Why character-level classification?

Character-level modelling is useful for names because:

- Names can have many spelling variations.
- Indian names may originate from different languages and regions.
- The same name can appear in different transliterations.
- Character patterns can capture prefixes, suffixes and internal structures.
