# 🤖 Machine learning dashboard

This is a dashboard to display insights and predictionns from a machine learning model

# Key Features
## 1.Interactive UI with Streamlit
   User-friendly interface built with Streamlit for easy interaction.
    Sidebar sliders for setting input parameters like Sepal and Petal dimensions.

## 2. Data Exploration
Raw data display with an expandable section to view the Iris dataset.
 Visualization of feature columns (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm) and target (Species).

## 3. Machine Learning Models
  Multiple models available for prediction:
    Logistic Regression (LR)
    Linear Discriminant Analysis (IDA)
    K-Nearest Neighbors (KNN)
    Decision Tree (CART)
    Gaussian Naive Bayes (NB)
    Support Vector Machine (SVM)
  Automatic model evaluation and selection based on cross-validation accuracy.

## 4. Model Evaluation
  Training and testing split with 80/20 ratio.
  Cross-validation applied to each model for performance comparison.
  Displays model accuracy metrics.

## 5. Real-Time Predictions
   Predicts the species of Iris flowers based on user-defined input features.
   Displays the predicted species in real-time.

## 6. Extensible Design
  Easy to extend with additional models or datasets.
  Modular structure allows for quick updates and customization.

## Usage
Adjust the input parameters using the sidebar sliders.
    View the data and selected machine learning model.
    Observe the prediction output displayed below the input parameters.

## Future Enhancements

 Model Performance Visualization: Add graphical representations of model performance (e.g., confusion matrix, ROC curves).
    Custom Dataset Upload: Allow users to upload their datasets for prediction.
    Advanced Hyperparameter Tuning: Implement grid search or randomized search for model optimization.
    Improved UX: Enhance the interface with more interactive data visualizations.

## Conclusion
he Machine Learning Prediction Dashboard is a robust and interactive application designed to simplify the process of model evaluation and prediction using the Iris dataset. By integrating multiple machine learning models and providing a user-friendly interface, this project demonstrates the power of Streamlit in creating accessible data science tools. Users can easily explore data, adjust input features, and obtain real-time predictions, making it a valuable resource for both learning and practical applications in machine learning.

This project not only showcases the capabilities of various classification models but also highlights the importance of user interaction in understanding and interpreting machine learning results. With its extensible design, the dashboard serves as a strong foundation for further enhancements, such as incorporating additional datasets, advanced model tuning, and improved data visualizations, ultimately contributing to a more comprehensive data science experience.
## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://Streamlit-ML-model-dash.streamlit.app/)

## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/app-starter-kit?quickstart=1)


