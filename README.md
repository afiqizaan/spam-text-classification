# Spam Text Classification

This project implements a machine learning pipeline to classify text messages as spam or ham (not spam). 
It utilizes various datasets, hyperparameter tuning, and neural network models to achieve robust classification performance.

## Project Structure
```bash
├── HyperparameterTuning.ipynb  # Notebook for optimizing model parameters
├── NeuralNetwork_gmail_fadzwan.ipynb  # Notebook for implementing a neural network on email data
├── SMS_Spam.ipynb              # Notebook for classifying SMS messages as spam or ham
├── gmail-fadzwan - 1.csv       # Dataset for email classification
├── spam.csv                    # SMS dataset for spam detection
├── spam_ham_dataset.csv        # Additional dataset for text classification
├── requirements.txt            # List of dependencies
└── README.md                   # Project documentation
```
.

## Project Details

This website showcases information about **BarBois** and serves as a digital resource for individuals interested
in calisthenics. Built with HTML and CSS, the site is designed to be simple, informative, and visually appealing, 
giving users a sense of the BarBois mission and activities.

## Features

- **Data Exploration**: Analyze datasets to understand the distribution of spam and ham messages.
- **Text Preprocessing**: Includes tokenization, stopword removal, stemming/lemmatization, and feature extraction using TF-IDF.
- **Machine Learning Models**: Logistic Regression, Naive Bayes, and SVM are used for classification.
- **Deep Learning Model**: A neural network is implemented for email classification.
- **Hyperparameter Tuning**: Optimizes model performance using techniques like GridSearchCV.
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix are evaluated.


## Datasets

1. **gmail-fadzwan - 1.csv**: Email data for neural network classification.
2. **spam.csv**: A collection of SMS messages labeled as spam or ham.
3. **spam_ham_dataset.csv**: A supplementary dataset for text classification tasks.

## Getting Started
### Prerequisites

- Python 3.7+
- Jupyter Notebook or a compatible environment.

### Installation
1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/spam-text-classification.git
cd spam-text-classification
```
2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```
3. **Install the required dependencies:**
```bash
pip install -r requirements.txt
```

## Running the Project
1. **Preprocessing and Feature Extraction:**
   - Run the notebooks (SMS_Spam.ipynb or NeuralNetwork_gmail_fadzwan.ipynb)
     to preprocess text data and extract features using TF-IDF or other methods.
2. **Model Training and Evaluation:**
   - Train different models and compare their performance. The training process is detailed in the provided notebooks.
3. **Hyperparameter Tuning:**
   - Use the HyperparameterTuning.ipynb notebook to optimize model parameters for better accuracy.
  
## Results
- **Model Performance:** Achieved high accuracy and robust metrics for spam detection.
- **Neural Network:** The neural network implementation effectively classifies email messages.

## Known Issues
- **Dataset Imbalance:** Some datasets might have a skewed distribution of spam vs. ham messages. Consider oversampling or undersampling for better results.
- **Text Lengths:** Extremely long or short texts may affect feature extraction and model performance.

## Credits
- **Datasets:** UCI ML Repository, Kaggle, and Gmail data.
- **Python Libraries:** Scikit-learn, TensorFlow/Keras, Pandas, NumPy, etc.
