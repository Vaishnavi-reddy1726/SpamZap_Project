# **SpamZap — Spam Email Classifier**

An end-to-end NLP pipeline that classifies emails and messages as **spam** or **ham (not spam)** using NLTK for text preprocessing, TF-IDF for feature extraction, and a Multinomial Naive Bayes classifier with Laplace smoothing.

**Test-Set Performance:**

- **Accuracy:** ~98%
- **Precision:** ~97%
- **Spam Recall:** ~87%

## **How It Works**

```text
Raw Message
     │
     ▼
clean_text()
Lowercase • Remove URLs • Emails • Punctuation • Digits
     │
     ▼
tokenize_and_stem()
NLTK Tokenization → Stopword Removal → Porter Stemming
     │
     ▼
TF-IDF Vectorizer
Unigrams + Bigrams • Up to 5,000 Features
     │
     ▼
Multinomial Naive Bayes
Alpha = 0.1 • Laplace/Additive Smoothing
     │
     ▼
Spam / Ham + Prediction Probability
```

## **Text Preprocessing**

`preprocess.py` processes raw messages through:

- Lowercasing
- URL removal
- Email removal
- Punctuation removal
- Digit removal
- NLTK tokenization
- Stopword removal
- Porter stemming

The processed text is then converted into numerical features using TF-IDF.

## **Feature Extraction**

The model uses a TF-IDF vectorizer with:

- Unigrams
- Bigrams
- Maximum 5,000 features

This allows the classifier to learn important words and short phrases commonly associated with spam.

## **Spam Classification**

A `MultinomialNB` classifier is trained using:

```text
alpha = 0.1
```

Laplace/additive smoothing helps the model handle words that were not observed during training.

The model outputs:

- **SPAM** or **HAM**
- Spam probability

## **Project Structure**

```text
spamzap/
│
├── data/
│   └── sms.tsv
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   ├── spamzap_model.joblib
│   └── tfidf_vectorizer.joblib
│
├── outputs/
│   ├── metrics.json
│   ├── classification_report.txt
│   └── confusion_matrix.png
│
├── main.py
├── requirements.txt
└── README.md
```

## **Setup**

Install the required dependencies:

```bash
pip install -r requirements.txt
```

On the first run, the required NLTK resources are automatically downloaded if they are not already installed:

- `punkt`
- `stopwords`

## **Usage**

### **1. Train the Model**

Train the classifier using the dataset:

```bash
python main.py train
```

Or run the training script directly:

```bash
python src/train.py
```

The training pipeline:

1. Loads the dataset.
2. Cleans and preprocesses messages.
3. Converts text into TF-IDF features.
4. Splits the dataset into training and test sets.
5. Trains the Multinomial Naive Bayes classifier.
6. Evaluates model performance.
7. Saves the trained model and vectorizer.

### **2. Classify a Message**

Run:

```bash
python main.py predict "Congratulations! You've won a free cruise, call now!"
```

Example output:

```text
Label     : SPAM
Spam Prob : 96.4%
```

### **3. Interactive Mode**

Run:

```bash
python main.py predict -i
```

You can then enter messages interactively for classification.

## **Dataset**

The project uses the **SMS Spam Collection dataset**, containing:

- **5,572 labeled messages**
- Spam and ham classifications
- Approximately **13% spam messages**

Dataset format:

```text
label<TAB>message
```

Example:

```text
ham	Hey, are we still meeting today?
spam	Congratulations! You have won a free prize. Call now!
```

To use your own dataset, replace:

```text
data/sms.tsv
```

with a tab-separated file following the same format.

## **Evaluation Methodology**

The dataset is divided using an **80/20 stratified train-test split** to preserve the spam-to-ham ratio across both sets.

### **Accuracy**

Measures the overall percentage of correctly classified messages.

### **Precision**

Of all messages predicted as spam, precision measures how many were actually spam.

Higher precision reduces false positives, preventing legitimate messages from being incorrectly classified as spam.

### **Recall**

Measures how many actual spam messages were successfully detected.

Higher recall reduces false negatives, preventing spam from reaching the user.

### **Confusion Matrix**

The model also generates a confusion matrix showing:

- True Positives
- False Positives
- True Negatives
- False Negatives

The output is saved as:

```text
outputs/confusion_matrix.png
```

## **Outputs**

After training, the following files are generated:

```text
models/
├── spamzap_model.joblib
└── tfidf_vectorizer.joblib

outputs/
├── metrics.json
├── classification_report.txt
└── confusion_matrix.png
```

## **Future Improvements**

- Replace `MultinomialNB` with `ComplementNB`.
- Compare performance with a Linear SVM.
- Experiment with class balancing to improve spam recall.
- Add oversampling for the minority spam class.
- Build a Flask or FastAPI service for real-time predictions.
- Add a web interface for message classification.

## **Technologies Used**

- Python
- NLTK
- Scikit-learn
- Pandas
- NumPy
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Natural Language Processing
- Porter Stemming
