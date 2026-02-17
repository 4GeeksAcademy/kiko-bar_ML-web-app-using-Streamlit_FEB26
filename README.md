# California Housing Cluster Predictor

A machine learning application that classifies California neighborhoods into geographic-economic profiles using **Supervised** and **Unsupervised Learning**.

**Live App:** https://kiko-bar-ml-web-app-using-streamlit-feb26.onrender.com

## Project Overview
This project analyzes the California Housing dataset to identify distinct neighborhood "clusters" based on location and income. By excluding high-income outliers (>$50k), the model provides a more accurate representation of the majority housing market.

### The Methodology
1. **Unsupervised Learning:** Used K-Means clustering to identify 6 distinct geographic-economic zones (Target Labels).
2. **Supervised Learning:** Trained a Random Forest Classifier to predict these clusters using only three key features: **Latitude**, **Longitude**, and **Median Income**.
3. **Data Engineering:** Developed a "Bridge" pipeline to handle 5-feature scaling for a 3-feature prediction model.

## Project Structure
```text
.
├── data/               # Raw and processed datasets
├── models/             # Trained .pkl models and scalers
├── src/
│   ├── app.py          # Streamlit user interface
│   └── utils.py        # Helper functions
└── requirements.txt    # Dependency list for Render
