# Real-Time Disaster Detection on Social Media

> A machine learning-based system for classifying disaster-related tweets in real time, aiding emergency responders and humanitarian efforts.

## 📂 Dataset Overview

**Source**: `train.csv`  
Contains approximately **7,600 tweets** with the following columns:

- `id`: Unique identifier for each tweet
- `text`: The actual tweet content
- `location`: Origin of the tweet (may be blank)
- `keyword`: Keyword associated with the tweet (may be blank)
- `target`: Binary label (1 = disaster, 0 = not disaster)

---

## 🚨 Problem Statement

Social media platforms like **Twitter** are crucial during natural disasters. However, the informal nature of tweets, metaphorical usage of disaster terms (e.g., *"I'm ablaze with happiness"*) and an overwhelming amount of non-disaster content pose serious challenges in filtering relevant tweets for emergency responders.

---

## 💡 Proposed Solution

We present two machine learning pipelines:

### 1. **Traditional Pipeline (TF-IDF + Classical Models)**
- **Preprocessing**: Emoji & URL removal, text normalization, keyword lemmatization
- **Features**: TF-IDF vectors
- **Models**: Logistic Regression, Multinomial Naive Bayes, Random Forest

### 2. **Advanced Pipeline (BERT + XGBoost)**
- **Embeddings**: Contextual BERT embeddings from `bert-base-uncased`
- **Classifier**: XGBoost
- **Advantage**: Better semantic understanding, handles ambiguity

---

## 📈 Results

| Model                    | Accuracy | F1 Score |
|-------------------------|----------|----------|
| Logistic Regression     | 0.7859   | 0.7371   |
| Multinomial Naive Bayes | 0.7892   | 0.7323   |
| Random Forest           | 0.7754   | 0.7159   |
| **BERT + XGBoost**      | **0.8207** | **0.7760** |

---

## 🌐 Web UI (Prototype)
A responsive **web interface** was created using **Lovable** AI tool, allowing users to enter a tweet and get instant classification:

🔗 [Try the App](https://alert-ai-response-team.lovable.app/)

---

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `disaster_project_1.py` | Implements traditional ML pipeline using TF-IDF and models like Logistic Regression, Naive Bayes, and Random Forest. |
| `disaster_project_2(updated).py` | Advanced pipeline using BERT embeddings and XGBoost for superior contextual classification. |
| `Disaster_Report.pdf` | Final report including problem, methodology, experiments, and conclusions. |

---

## 🔧 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Prachikatare/disaster-tweet-detector.git
   cd disaster-tweet-detector
