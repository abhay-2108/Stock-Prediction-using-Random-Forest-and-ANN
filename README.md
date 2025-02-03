# Stock Price Prediction using Machine Learning

## Overview
This project aims to predict stock prices using machine learning techniques. It involves data preprocessing, exploratory data analysis (EDA), model building using traditional machine learning algorithms and deep learning techniques, and trend visualization.

## Dataset
The dataset used in this project contains historical stock price data, including features such as Open, High, Low, Volume, and Adjusted Close Price. The data was cleaned, missing values were handled, and exploratory analysis was conducted to understand correlations between variables.

## Features
- **Open Price**: Stock price at market open
- **High Price**: Highest price during the trading day
- **Low Price**: Lowest price during the trading day
- **Volume**: Number of shares traded
- **Adj Close Price**: Adjusted closing price (target variable)

## Algorithms Used
This project employs multiple machine learning algorithms for stock price prediction:

### 1. **Random Forest Regressor**
- Used with GridSearchCV for hyperparameter tuning
- **Performance Metrics:**
  - Mean Squared Error (MSE): *Calculated during evaluation*
  - Mean Absolute Error (MAE): *Calculated during evaluation*
  - R2 Score: *Calculated during evaluation*

### 2. **Artificial Neural Network (ANN) using TensorFlow/Keras**
- A deep learning approach with multiple hidden layers
- Architecture:
  - Input Layer: 128 neurons, ReLU activation
  - Hidden Layers: 64, 32, 16, 8 neurons with ReLU activation
  - Output Layer: 1 neuron (linear activation)
- **Performance Metrics:**
  - Mean Absolute Error (MAE) on Test Set: *Calculated during evaluation*
  - R2 Score: *Calculated during evaluation*

## Model Comparison
| Model | Mean Squared Error | Mean Absolute Error | R2 Score |
|--------|-------------------|--------------------|-----------|
| Random Forest Regressor | *0.1050* | *0.1668* | *0.99942* |
| ANN (Deep Learning) | *-* | *-* | *0.9986* |

> The ANN model with multiple hidden layers performed well in capturing complex patterns in stock prices.

## Results & Visualization
- The actual vs. predicted stock prices were plotted to observe how well the models captured trends.
- Upward and downward price trends were detected and visualized using color-coded annotations.
- A heatmap was generated to analyze feature correlations.

## How It Works
1. **Data Preprocessing**:
   - Removed duplicate values
   - Handled missing data
   - Feature selection (removed 'Date')
2. **Exploratory Data Analysis (EDA)**:
   - Data visualization using histograms and scatter plots
   - Correlation matrix and heatmap analysis
3. **Model Training & Evaluation**:
   - Train-test split (80-20 ratio)
   - Training using Random Forest and ANN
   - Hyperparameter tuning for optimal performance
4. **Stock Trend Prediction**:
   - Trend classification based on prediction differences
   - Plotted stock trends with annotations

## Installation & Dependencies
To run this project, install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```

