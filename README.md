# nlp4climate - NLP Life Expectancy & Climate Change 🌍🧠

This project analyzes academic, institutional and media texts related to life expectancy and climate change. It uses NLP techniques such as WordClouds, LDA and supervised classification to identify thematic and stylistic differences across types of sources.

The app is available in **two languages**: Catalan and German.

---


## 🔧 Requirements

- Python 3.9+
- The libraries listed in `requirements.txt`

---


## 📁 Project structure

```
nlp4climate/
│
├── Data/                # Corpus and generated models (LDA, dictionaries, etc.)
├── models/              # Trained models required for classification
├── Scripts/
│   ├── main.py          # Main Streamlit app script
│   └── NLP_functions.py # Utility functions for loading data and visualization
│   └── content/
│   ├── ca_content.py    # App content in Catalan
│   └── de_content.py    # App content in German
├── requirements.txt     # Library list with versions
└── README.md            # This file
└── LICENSE.txt          # License (MIT) of the project
```

## ▶️ How to run the project

1. Clone this repository:

```bash
git clone https://github.com/abenet25/nlp4climate.git
cd nlp4climate
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the Streamlit app:

```bash
streamlit run Scripts/Main.py
```

## ⚠️ Important
The app relies on pre-trained models stored in the models/ folder (vectorizer.pkl, logistic_regression.pkl, naive_bayes.pkl). These files must not be modified or removed, as they are essential for correct classification.

## ✨ Features

- Multilingual interface: 🇩🇪 German and Catalan
- WordCloud visualizations (general and topic-specific with LDA)
- Topic bar charts per corpus (academic, institutional, media)
- Supervised classification of sentences by origin (Naive Bayes, Logistic Regression)

- WordCloud visualizations for each corpus
- Topic modeling (LDA) with topic-specific bar charts
- Sentence classification by text origin:
        Academic 🧑‍🎓
        Institutional 🏛️
        Media 📰
- Comparison of classification performance: Multinomial Naive Bayes vs Logistic Regression

## 👤 Author

Ariadna Benet. Project developed as part of a final project during the Data Science Bootcamp by Data Science Institute.
https://github.com/abenet25/

## 📄 License

This project is licensed under the [MIT License](LICENSE).

# 
