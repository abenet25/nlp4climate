import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from gensim.models import LdaModel
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from NLP_functions import (
    load_dict_and_corpus, 
    generate_wordcloud, 
    generate_topic_wordcloud,
    generate_topic_bar_chart,
    plot_word_vectors
)


# Dictionarys und Corpus den drei Korpora: 
corpus_paths = {
    "Wissenschaft": {
        "dict": "Data\lda_dictionary_academia.dict",
        "corpus": "Data\lda_corpus_academia.mm"
    },
    "Institutionen": {
        "dict": "Data\lda_dictionary_institutions.dict",
        "corpus": "Data\lda_corpus_institutions.mm"
    }, 
    "Medien": {
        "dict": "Data\lda_dictionary_media.dict",
        "corpus": "Data\lda_corpus_media.mm"
    }
}

# lda_models:
lda_models = {
    "Wissenschaft": LdaModel.load("Data/lda_models_academia.model"),
    "Institutionen": LdaModel.load("Data/lda_models_institutions.model"),
    "Medien": LdaModel.load("Data/lda_models_media.model")
}

st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] {
            background-color: darkblue;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Titel
st.title("Lebenserwartung und Klimawandel: NLP Analyse")
st.subheader("Word Clouds, Themenvisualisierung, Satzklassifikation")

# Seitenmenü (Sidebar)
st.sidebar.title("🧭 Navigation")
option = st.sidebar.radio(
    "Wähle eine Sektion:",
    ("📄 Projektbeschreibung", 
     "📲 Daten und Methodologie", 
     "🗂️ Themenvisualisierung", 
     "🧠 Satzklassifikation")
)

# Projektbeschreibung
if option == "📄 Projektbeschreibung":
    st.title("📄Projektbeschreibung")
    st.markdown("""
    Willkommen zu unserem Natural Language Processing (NLP) Projekt!
            
    In diesem Projekt analysieren wir englischsprachige Texte über Lebenserwartung und Klimmawandel aus drei verschiedenen Quellen:
    - 👩‍🏫 Wissenschaftliche Publikationen  
    - 🏛️ Institutionelle Berichte  
    - 📰 Medienartikel  

    Das Hauptziel des Projekts ist es herauszufinden, ob und in welche Richtung die Auswirkungen des Klimawandels die Lebenserwartung beeinflussen.  
    Da nummerische Datensätze beiden Feldern schwer zu verbinden sind, versuchen wir,  
    Texte mit ML zu explorieren und daraus Antworten auf unsere Fragen zu gewinnen.  
    Dazu untersuchen wir die häufigsten Themen der drei Akteure (Wissenschaft, Institutionen und Medien)  
    und können durch einen Satzklassifikator vorhersagen, aus welcher Quelle ein bestimmter Satz stammt.
    """)

# Daten und Methodologie
elif option == "📲 Daten und Methodologie":
    st.header("📲 Daten und Methodologie")
    # Daten
    st.subheader("🔍 Datenübersicht")
    st.markdown("""
    Die Korpora wurden mit API / manuell zusammengestellt und bereinigt. Insgesamt analysieren wir:

    - 432 wissenschaftliche Abstracts (API Semantic Scholar)
    - 22 institutionelle Berichte (WHO, UN, World Bank)
    - 29 journalistische Artikel (BBC, Euronews, Reuters, The Conversation, u.a.)

    Suchwörter: "life expectancy" AND "climate change"; "health" AND "climate change"

    Wir arbeiten mit insgesamt über **154.000 Wörtern**.
    """)
 
    # Daten
    labels = ["Wissenschaft", "Institutionen", "Media"]
    sizes = [96909, 29716, 27720]
    colors = ["#66b3ff", "#99ff99", "#ffcc99"]

    # Prozent und absolute Werte
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return f"{pct:.1f}%\n({val:,} Wörter)"
        return my_autopct

    # Pie chart
    fig, ax = plt.subplots(figsize=(3, 3))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=make_autopct(sizes),
        startangle=140,
        colors=colors,
        textprops={'fontsize': 7}
    )

    # Größe der Beschreibung
    for text in texts:
        text.set_fontsize(8)
    for autotext in autotexts:
        autotext.set_fontsize(6)
    ax.axis("equal")  # Sichert runde Form

    st.markdown("### Verteilung der Korpora nach Quellen")
    st.pyplot(fig)
    st.markdown(f"**Gesamtanzahl der Wörter:** {sum(sizes):,}")

    # Methoden
    st.subheader("⚙️ Methoden & Vorgehen")
    st.markdown("""
    Folgende Schritte wurden durchgeführt:

    1. **Preprocessing**: Kleinschreibung, Sonderzeichen- und Trennlinienentfernung.  
    Außerdem für jeden Teil spezifisch: 

    | Modul           | Stopwords                                 | Bigrams | Tokenisierung | Analyse-Einheit | Label |
    |----------------|--------------------------------------------|---------|----------------|------------------|--------|
    | **Word Cloud** | *english*, Schlüsselwörter, korpusspezifisch | ✅      | ✅             | Wort             | ❌     |
    | **LDA**        | *english*, Schlüsselwörter, korpusspezifisch | ✅      | ✅             | Wort             | ❌     |
    | **Word2Vec**   | *english*, korpusspezifisch                | ✅      | ✅             | Wort             | ❌     |
    | **Klassifikation** | keine                                  | ❌    | Text > Sätze              | Satz             | ✅     |
    
    **Bigram**: z.B. "life_expectancy"  
    **Tokenisierung**: Text wird in kleinere Einheiten zerlegt, hier: Wörter.
    
    2. **Unsupervised Learning**:  
    - **Word Cloud** pro Korpus  
    - **LDA-Themenmodellierung** für Themenvisualisierung
    - **Word2Vec** auf gesamten Korpus
    3. **Supervised Learning**:  
    - Klassifikation von Sätzen: **Multinomial Naive Bayes** vgl. **Logic Regression**
    - Nutzer*in kann einen Satz eingeben und erhält die **wahrscheinlichste Quelle** (Wissenschaft, Institution oder Medien)
    """)



# Seite: Themenvisualisierung
elif option == "🗂️ Themenvisualisierung":
    st.title("🗂️ Themenvisualisierung")
    st.markdown("""
    Hier zeigen wir die wichtigsten Themen jedes Korpus mit:
    
    - 🔠 **Word Clouds**  
    - 📊 **LDA-Topic modelling** 
    - 📐 **Word2Vec**
    """)

    st.subheader("🔠 **Word Clouds**")
 
    # Auswahl des Korpuses
    corpus_choice = st.selectbox(
        "Wähle einen Korpus aus:",
        list(lda_models.keys()),
        key="lda_corpus"
    )

    # Load Wörterbuch und Korpus
    path_dict = corpus_paths[corpus_choice]["dict"]
    path_corpus = corpus_paths[corpus_choice]["corpus"]
    dictionary, corpus = load_dict_and_corpus(path_dict, path_corpus)

    # LDA model corresponent
    lda_model = lda_models[corpus_choice]

    # Word Cloud generieren und zeigen
    with st.expander("Word Cloud des gesamten Korpus", expanded=True):
        wc = generate_wordcloud(dictionary, corpus)
        st.subheader(f"Word Cloud des gesamten Korpus: {corpus_choice}")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    # Funktion zur Generierung der Thema-Bezeichnungen
    def get_topic_labels(lda_model, topn=1):
        labels = []
        for i in range(lda_model.num_topics):
            top_words = lda_model.show_topic(i, topn=topn)
            main_word = top_words[0][0]
            labels.append((i, f"Thema {i+1} – {main_word}"))
        return labels
    
    # Themenliste aufbauen
    topic_labels = get_topic_labels(lda_model)
    label_to_id = {label: idx for idx, label in topic_labels}

    st.subheader("📊 **LDA-Topic modelling**" )

    # Thema auswählen
    topic_label = st.selectbox(f"5 Top-Themen im Korpus {corpus_choice}", list(label_to_id.keys()), key="lda_topic")
    topic_id = label_to_id[topic_label]

    col1, col2 = st.columns(2)
    with col1:
        wc_topic = generate_topic_wordcloud(lda_model, topic_id)
        fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
        ax_wc.imshow(wc_topic, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
    with col2:
        fig_bar = generate_topic_bar_chart(lda_model, topic_id)
        st.pyplot(fig_bar)
    
    st.markdown("""
    Diese Themen wurden mit einem LDA-Modell aus dem Korpus extrahiert.  
    Jedes Thema besteht aus Wörtern, die häufig zusammen im gleichen Kontext auftreten.  
    LDA: "Latent Dirichlet Allocation".
    """)    

    st.subheader("📐 **Word2Vec:** Semantische Wortbeziehungen entdecken")

    st.markdown("""
    **Word2Vec** ist ein Modell, das Wörter als numerische Vektoren darstellt, 
    basierend auf ihren semantischen Kontexten im Textkorpus.
    Dadurch lassen sich Beziehungen zwischen Begriffen identifizieren 
    und in einem zweidimensionalen Raum visualisieren.
    
    📍 In der folgenden Grafik sieht man eine PCA-Projektion dieser Vektoren.
    Je näher zwei Punkte beieinander liegen, desto ähnlicher wurden sie vom Modell verstanden.
    
    """)

    # Modell laden
    @st.cache_resource
    def load_model():
        return Word2Vec.load("models/word2vec_global.model")
    
    model_w2v = load_model()

    # 🌐 PCA-Visualisierung

    input_words = st.text_input(
        "Wörter zum Visualisieren (durch Leerzeichen getrennt):",
        "climate_change life_expectancy health air_pollution policy"
    )

    words_list = input_words.strip().split()
    words_checked = [w.lower().strip() for w in input_words.strip().split()]
    words_found = [w for w in words_checked if w in model_w2v.wv]

    if words_found:
        fig = plot_word_vectors(model_w2v, words_found)
        if fig:
            st.pyplot(fig)
    else:
        st.error("⚠️ Keine der eingegebenen Wörter ist im Modell vorhanden.")

   


# Seite: Satzklassifikation
elif option == "🧠 Satzklassifikation":
    st.title("🧠 Klassifizierung von Sätzen")
    st.markdown("""
    Gib unten einen beliebigen Satz über Lebenserwartung und Klimmawandel auf Englisch ein und unser Modell sagt dir voraus,
    ob der aus einem **wissenschaftlichen**, **institutionellen** oder **journalistischen** Text stammt.
    """)

    # Lade Modell und Vektorisierer
    @st.cache_resource
    def load_model():
        base_path = os.path.dirname(os.path.abspath(__file__))  
        model_dir = os.path.join(base_path, "..", "models")
        with open(os.path.join(model_dir, "vectorizer.pkl"), "rb") as f1, \
            open(os.path.join(model_dir, "logistic_regression.pkl"), "rb") as f2, \
            open(os.path.join(model_dir, "naive_bayes.pkl"), "rb") as f3:
            return pickle.load(f1), pickle.load(f2), pickle.load(f3)

    vectorizer, model_lr, model_nb = load_model()

    try:
        check_is_fitted(vectorizer, "idf_")
    except NotFittedError:
        st.error("❌ Der Vektorisierer wurde nicht trainiert. Bitte überprüfe die Datei 'vectorizer.pkl'.")
        st.stop()
        
    # Benutzereingabe
    user_input = st.text_input("✏️ Satz eingeben:")

    if user_input:
        # In Vektoren umwandeln und vorhersagen
        X_input = vectorizer.transform([user_input])
        pred = model_lr.predict(X_input)[0]
        probas = model_lr.predict_proba(X_input)[0]
        classes = model_lr.classes_

        st.success(f"Modellvorhersage: **{pred}**")

        # Wahrscheinlichkeiten anzeigen
        st.markdown("### 🎲 Wahrscheinlichkeiten:")
        for label, prob in zip(classes, probas):
            st.write(f"• **{label}**: {prob * 100:.1f}%")
        
        # Barchart der Wahrscheinlichkeit
        proba_df = pd.DataFrame({"Klasse": classes, "Wahrscheinlichkeit": probas})
        proba_df = proba_df.sort_values("Wahrscheinlichkeit", ascending=True)

        fig, ax = plt.subplots(figsize=(3, 1.2))
        ax.barh(proba_df["Klasse"], proba_df["Wahrscheinlichkeit"], color="skyblue")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Wahrscheinlichkeit", fontsize=4)
        ax.set_title("Verteilung der Klassifikation", fontsize = 5)
        ax.tick_params(axis='both', labelsize=5)
        fig.tight_layout(pad=1)
        st.pyplot(fig)

        st.subheader(" 🤖 Multinomial Naive Bayes vs Logistic Regression")

        with open("models/naive_bayes.pkl", "rb") as f:
            nb_model = pickle.load(f)

        with open("models/logistic_regression_tr.pkl", "rb") as f:
            lr_model = pickle.load(f)

        with open("models/X_test_tfidf.pkl", "rb") as f1, open("models/y_test.pkl", "rb") as f2:
            X_test_tfidf = pickle.load(f1)
            y_test = pickle.load(f2)

        
        nb_preds = nb_model.predict(X_test_tfidf)
        lr_preds = lr_model.predict(X_test_tfidf)       

        def plot_confusion_matrix(y_true, y_pred, labels, title):
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            fig, ax = plt.subplots(figsize=(2.5, 2.5))
            heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels, 
                        annot_kws={"size": 6}, ax=ax, 
                        cbar_kws={"shrink": 0.8})
            cbar = heatmap.collections[0].colorbar
            cbar.ax.tick_params(labelsize=6)

            ax.set_xlabel("Predicted", fontsize=6)
            ax.set_ylabel("Actual", fontsize=6)
            ax.tick_params(axis='both', labelsize=4)
            fig.tight_layout(pad=1)
            st.pyplot(fig)
        
        

        st.subheader("🔹 Confusion Matrix Multinomial Naive Bayes")
        plot_confusion_matrix(y_test, nb_preds, labels=nb_model.classes_, title="Confusion Matrix – Naive Bayes")

        st.subheader("🔹 Confusion Matrix Logistic Regression")
        plot_confusion_matrix(y_test, lr_preds, labels=lr_model.classes_, title="Confusion Matrix – Logistic Regression")


        st.markdown("## 📝 Classification Reports")


        
        def classification_report_to_df(report_str):
            lines = report_str.strip().split("\n")
            data = []
            for line in lines[2:-3]:
                parts = line.strip().split()
                if len(parts) == 5:
                    label = parts[0]
                    precision, recall, f1, support = map(float, parts[1:])
                    data.append((label, precision, recall, f1, int(support)))
            return pd.DataFrame(data, columns=["Label", "Precision", "Recall", "F1-Score", "Support"])

        def style_report_table(df):
            return (
                df.style
                .format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-Score": "{:.2f}"})
                .set_properties(subset=["Precision", "Recall", "F1-Score", "Support"], **{"text-align": "right"})
                .set_table_styles([
                    {"selector": "th", "props": [("text-align", "right")]},  # Alineació de capçaleres
                ])
            )

        report_nb_str = classification_report(y_test, nb_preds, target_names=model_nb.classes_)
        report_lr_str = classification_report(y_test, lr_preds, target_names=lr_model.classes_)

        report_nb_df = classification_report_to_df(report_nb_str)
        report_lr_df = classification_report_to_df(report_lr_str)

        # Calcular l'accuracy manualment
        accuracy_nb = accuracy_score(y_test, nb_preds)
        accuracy_lr = accuracy_score(y_test, lr_preds)

        st.markdown(f"➡️ **Accuracy – Naive Bayes**: {accuracy_nb:.2%}")
        st.markdown(f"➡️ **Accuracy – Logistic Regression**: {accuracy_lr:.2%} 🏅")

        st.subheader("🔹 Multinomial Naive Bayes")
        st.dataframe(style_report_table(report_nb_df))

        st.subheader("🔹 Logistic Regression")
        st.dataframe(style_report_table(report_lr_df))

        
