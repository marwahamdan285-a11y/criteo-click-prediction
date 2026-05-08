# Ad Click Prediction — Criteo Terabyte Dataset

A Streamlit web app that predicts whether a user will click on an advertisement using a **Gradient Boosting** model trained on the Criteo Terabyte dataset sample.

## Project Structure

```
├── app.py               # Streamlit application
├── requirements.txt     # Python dependencies
└── README.md
```

## Features

- Predicts ad click probability from user and ad features
- Built with Gradient Boosting (best F1-Score: 0.4088)
- Interactive sidebar inputs for all features
- Real-time prediction with probability score

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

## Models Compared

| Model | Accuracy | F1-Score |
|---|---|---|
| Gradient Boosting | 0.7602 | 0.4088 ✅ |
| CatBoost | 0.7593 | 0.3940 |
| XGBoost | 0.7316 | 0.3884 |
| Logistic Regression | 0.7866 | 0.3778 |
| LightGBM | 0.7207 | 0.3771 |
| Random Forest | 0.7651 | 0.3730 |
| KNN | 0.7389 | 0.3701 |

## Dataset

[Criteo Terabyte Click Logs — Kaggle](https://www.kaggle.com/c/criteo-display-ad-challenge)

## Tech Stack

- Python, Scikit-learn, XGBoost, LightGBM, CatBoost
- Streamlit for deployment
- SMOTE for class imbalance
- SelectKBest for feature selection
