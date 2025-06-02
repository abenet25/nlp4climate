import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import pandas as pd
from pathlib import Path
from gensim.models import LdaModel, Word2Vec
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

# 📂 Base path = wo main.py wird durchgeführt
BASE_PATH = Path().resolve()
DATA_PATH = BASE_PATH / "Data"
MODELS_PATH = BASE_PATH / "models"


# Datenstrukturen laden
corpus_paths = {
    "Wissenschaft": {
        "dict": str(DATA_PATH / "lda_dictionary_academia.dict"),
        "corpus": str(DATA_PATH / "lda_corpus_academia.mm")
    },
    "Institutionen": {
        "dict": str(DATA_PATH / "lda_dictionary_institutions.dict"),
        "corpus": str(DATA_PATH / "lda_corpus_institutions.mm")
    }, 
    "Medien": {
        "dict": str(DATA_PATH / "lda_dictionary_media.dict"),
        "corpus": str(DATA_PATH / "lda_corpus_media.mm")
    }
}

lda_models = {
    "Wissenschaft": LdaModel.load(str(DATA_PATH / "lda_models_academia.model")),
    "Institutionen": LdaModel.load(str(DATA_PATH / "lda_models_institutions.model")),
    "Medien": LdaModel.load(str(DATA_PATH / "lda_models_media.model"))
}

# Klassifikator: Vectorizer und Modelle
@st.cache_resource
def load_classification_models():
    with open(MODELS_PATH / "vectorizer.pkl", "rb") as f1, \
         open(MODELS_PATH / "logistic_regression.pkl", "rb") as f2, \
         open(MODELS_PATH / "naive_bayes.pkl", "rb") as f3:
        return pickle.load(f1), pickle.load(f2), pickle.load(f3)


def show_de_content():
    # Titel
    st.markdown("<h3 style='margin-bottom: 0;'>🌍 Lebenserwartung und Klimawandel mit NLP</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin-top: 0;'>Word-Clouds, Themenvisualisierung, Satzklassifikation</h4>", unsafe_allow_html=True)

    # Navigation
    st.sidebar.title("🧭 Navigation")
    option = st.sidebar.radio(
        "Wähle eine Sektion:",
        ("📘Projektbeschreibung", 
         "📲 Daten und Methodologie", 
         "🗂️ Themenvisualisierung", 
         "🧠 Satzklassifikation", 
         "🚀 Verbesserungspotenziale", 
         "📋 Über das Projekt")
    )

    if option == "📘Projektbeschreibung":
        show_project_description()
    elif option == "📲 Daten und Methodologie":
        show_data_methodology()
    elif option == "🗂️ Themenvisualisierung":
        show_topic_visualization()
    elif option == "🧠 Satzklassifikation":
        show_sentence_classification()
    elif option == "🚀 Verbesserungspotenziale":
        st.title("🚀 Verbesserungspotenziale")
        st.markdown("""
        ↗️ **Institutionelle und journalistische Korpus zu erweitern**  
        Während zahlreiche akademische Texte bereits erfolgreich über eine API (Semantic Scholar) gesammelt wurden, 
        ist das Ziel, auf ähnliche Weise auch für institutionelle und journalistische Quellen geeignete Texte automatisiert 
        zu extrahieren – z.B. per API oder Web Scraping.    
        Dies würde die Datenbasis vergrößern und balancieren, und die Klassifikation verbessern. \n 
        👌 **Erhöhung der Sensitivität des Klassifikators**  
        Derzeit versucht das Modell, jeden Satz einer der drei vordefinierten Quellen (akademisch, institutionell oder journalistisch) zuzuordnen. 
        Eine mögliche Verbesserung wäre die Integration eines Mechanismus, der erkennt, ob der eingegebene Satz zum thematischen Bereich 
        des Gesamtkorpus gehört. Dies würde ermöglichen, **eine Vorhersage über den Grad der Plausibilität des Satzes im gegebenen Kontext** abzugeben 
        und zu warnen, wenn dessen Inhalt signifikant vom ursprünglichen Korpus abweicht. \n            
        🧱 **Verbesserung der Modularität der internen Projektstruktur**    
        Die Länge des Hauptscripts (de_content.py / ca_content.py) kann reduziert und besser verwaltet werden, 
        wenn die Funktionen in eigenständige Module ausgelagert werden.  
        Auch die Struktur der Daten (Modelle, Korpora und Wörterbücher) könnte neu organisiert werden,  
        um die Wartbarkeit und Skalierbarkeit zu verbessern.\n
        

        """)
    elif option == "📋 Über das Projekt":
        st.title("📋 Über das Projekt")
        st.markdown("""
        **Autorin:** Ariadna Benet  
        **Projekt durchgeführt im Rahmen des [Data Science Bootcamps – Data Science Institute by Fabian Rappert](https://www.data-science-institute.de)**  
        **Jahr:** 2025  

        📊 Datensatz als Ausgangspunkt: [Countries Life Expectancy (Kaggle)](https://www.kaggle.com/datasets/amirhosseinmirzaie/countries-life-expectancy)  
        📚 Bibliotheken: Streamlit, Gensim, scikit-learn, wordcloud, Matplotlib, NumPy, Pandas, Seaborn

        🔧 Dieses Projekt verwendet Techniken des Natural Language Processing (NLP), wie LDA, Word2Vec und supervised Klassifikatoren.
        """)

def show_project_description():
    st.markdown("<h2 style='margin-bottom: 0;'>📘Projektbeschreibung</h2>", unsafe_allow_html=True)
    st.markdown("""
    Willkommen zu unserem **Natural Language Processing (NLP) Projekt**!  
    Dieser Beitrag ist Teil des Abschlussprojekts **„Lebenserwartung: Analysen mit Business Intelligence, Machine Learning 
    und Natural Language Processing“**, durchgeführt im Rahmen des **Data Science Bootcamps** am [Data Science Institute by Fabian Rappert](https://www.data-science-institute.de).

    📊 **Datengrundlage**  
    Das Projekt stützt sich auf den Datensatz 👉 [Countries Life Expectancy](https://www.kaggle.com/datasets/amirhosseinmirzaie/countries-life-expectancy) *(Kaggle)*.
    Auf diesem Datensatz basieren die Analysen mit Business Intelligence und Machine Learning. Es wird untersucht, welche Variablen 
    die Lebenserwartung auf allen Kontinenten am stärksten beeinflussen. Außerdem werden Prognosen darüber erstellt, wie sich die Werte 
    in Abhängigkeit von Veränderungen der verschiedenen Faktoren entwickeln könnten.
                
    🌎 **Forschungsfrage**  
    Nachdem die Daten zur Lebenserwartung analysiert wurden, **stellt sich die Frage, ob der Klimawandel die Lebenserwartung beeinflusst 
    – oder beeinflussen könnte**.
    Welche Auswirkungen könnten globale Phänomene wie Temperaturanstieg oder Luftverschmutzung auf die Lebensdauer haben?
    Sind diese Effekte in reichen und armen Ländern gleich stark?
    Und haben eher lokale Folgen des Klimawandels – wie Naturkatastrophen – bereits erkennbare Spuren in der Lebenserwartung hinterlassen?
    Könnte es sein, dass entwickelte Länder künftig stärker vom Klimawandel betroffen sind als von bestimmten Krankheiten? 

    🔭 Ansatz    
    Da es schwierig ist, numerische Datensätze aus beiden Bereichen direkt miteinander zu verknüpfen, wählen wir einen alternativen Ansatz:
    Wir analysieren Texte zu beiden Themenbereichen mithilfe von Methoden des maschinellen Lernens, um unsere Forschungsfragen zu beantworten.
                
    🎯 **Zielsetzung**  
    Unser Ziel ist es, **das Thema Lebenserwartung mit dem des Klimawandels zu verknüpfen**, indem wir **englischsprachige Texte** 
    zu beiden Themen aus **drei unterschiedlichen Quellen** analysieren:

    - 👩‍🏫 Akademische Publikationen  
    - 🏛️ Institutionelle Berichte  
    - 📰 Journalistische Artikel  
    
    🗂️ Wir untersuchen die häufigsten Themen in Texten aus diesen drei Quellen.  
    🧠 Wir trainieren einen Satzklassifikator, der vorhersagt, aus welcher dieser drei Quellen ein gegebener Satz stammt.

    ⚙️ **Methodologie**  
    Durch **Themenmodellierung** (*Topic Modeling*), **semantische Wortdarstellungen** (*Word Embeddings*) und **Klassifikationsverfahren** 
    (mit *Multinomial Naive Bayes* und logistischer Regression) gewinnen wir Einblicke in die sprachliche Darstellung dieser globalen Themen.   
    Da numerische Datensätze aus beiden Bereichen schwer zu verknüpfen sind,  
    **wandeln wir Texte und Wörter in Vektoren und Zahlen um**.
                

    """, unsafe_allow_html=True)

def show_data_methodology():
    st.markdown("<h2 style='margin-bottom: 0;'>📲 Daten und Methodologie</h2>", unsafe_allow_html=True)
    # Daten
    st.subheader("🔍 Datenübersicht")
    st.markdown("""
    Die Korpora wurden mit API / manuell zusammengestellt und bereinigt. Insgesamt analysieren wir:

    - 432 wissenschaftliche Abstracts (API Semantic Scholar)
    - 22 institutionelle Berichte (WHO, UN, World Bank)
    - 29 journalistische Artikel (BBC, Euronews, Reuters, The Conversation, u.a.)
                
    Suchwörter: "life expectancy" AND "climate change"; "health" AND "climate change"
    
    Optimierungsziel: Zur besseren Ausbalancierung der drei Quellen sollen die institutionellen 
    und journalistischen Korpora erweitert werden –idealerweise durch automatisiertes Scraping 
    mithilfe von APIs (siehe 🚀 Verbesserungspotenziale). 

    Wir arbeiten mit insgesamt über **154.000 Wörtern**.
    """)
 
    # Daten
    labels = ["Wissenschaft", "Institutionen", "Media"]
    sizes = [96909, 29716, 27720]
    colors = ["#66b3ff", "#99ff99", "#ffcc99"]
    total = sum(sizes)

    # Bargraph
    fig, ax = plt.subplots(figsize=(4, 1.8))  

    bars = ax.barh(labels, sizes, color=colors)
    ax.set_xlabel("Anzahl der Wörter", fontsize=8)
    ax.set_title("Verteilung der Korpora nach Quellen", fontsize=8)

    # Werte neben die Bars
    for bar, count in zip(bars, sizes):
        percent = count / total * 100
        text = f"{count:,} ({percent:.1f}%)".replace(",", ".") 
        ax.text(count + total * 0.01, bar.get_y() + bar.get_height()/2,
                text, va='center', fontsize=7)

    # Format
    ax.tick_params(axis='both', labelsize=7)
    ax.set_xlim(0, max(sizes)*1.15)
    fig.tight_layout(pad=1)

    st.pyplot(fig)

    # Prozent und absolute Werte
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return f"{pct:.1f}%\n({val:,} Wörter)".replace(',', 'X').replace('.', ',').replace('X', '.')
        return my_autopct


    # Methoden
    st.subheader("⚙️ Methoden & Vorgehen")
    st.markdown("""
    Folgende Schritte wurden durchgeführt:

    1. **Preprocessing**: Kleinschreibung, Entfernung von Sonderzeichen und Trennlinien.
    Außerdem für jeden Teil spezifisch: 
    
        
    | Modul              | Tokenisierung     | Analyse-Einheit | Bigrams | Stopwords                                 | Label |
    |-------------------|-------------------|------------------|---------|--------------------------------------------|--------|
    | **Word Cloud**     | ✅                | Wort             | ✅      | Englisch, Schlüsselwörter, korpusspezifisch | ❌     |
    | **LDA**            | ✅                | Wort             | ✅      | Englisch, Schlüsselwörter, korpusspezifisch | ❌     |
    | **Word2Vec**       | ✅                | Wort             | ✅      | Englisch, korpusspezifisch                | ❌     |
    | **Klassifikation** | ✅                | Satz             | ❌      | keine                                     | ✅     |

    """)           
    with st.expander("✂️ Was heißt *Tokenisierung*?"):
        st.write("""**Tokenisierung** ist der erste Schritt in der automatisierten Textverarbeitung im Bereich des Natural Language Processing (NLP). 
                \nDabei werden Texte in kleinere Einheiten – sogenannte Tokens – zerlegt. Diese Einheiten, häufig Wörter oder Sätze, bilden die Grundlage für die weitere Analyse.
                 """)

    with st.expander("👯‍♀️ Was ist ein *Bigram*?"):
        st.write("""Ein **Bigram** ist ein häufig vorkommendes Paar von aufeinanderfolgenden Wörtern in einem Text. Zusammen bilden sie eine feste Bedeutungseinheit.
                 \n**Beispiel:** Das Wortpaar "air pollution" ist ein Bigram, da diese beiden Wörter oft zusammen verwendet werden und gemeinsam *Luftverschmutzung* bedeuten.
                """)
        
    with st.expander("🫸 Was sind *Stopwords*?"):
        st.write("""**Stopwörter** sind häufig vorkommende Wörter wie "und", "der" oder "ist", die meist keinen wichtigen Beitrag zum Inhalt eines Textes leisten und deshalb bei der Analyse oft weggelassen werden.
                 \n🔹**Enlisch:** Für jede Sprache werden spezifische Stoppwortlisten verwendet. Dabei handelt es sich meist um Funktionswörter, wie zum Beispiel im Englischen "the", "is", "and" usw.    
                🔹**Schlüsselwörter:** Spezifische Begriffe, die thematisch zentral, aber zu dominant sind – z.B. "health", "climate change" (Suchwörter bei der Textsammlung) – werden entfernt, um Verzerrung in WordClouds oder LDA-Themen zu vermeiden.    
                🔹**Korpusspezifisch:** Individuelle Liste basierend auf Häufigkeitsanalysen im jeweiligen Korpus – dient dazu, Wörter zu entfernen, die dort zwar oft vorkommen, aber analytisch wenig Mehrwert bieten. 
                 Für das wissenschaftliche Korpus, z.B. "study", "analysis" oder "conclusions".
                  """)

    st.markdown("""
    
    2. **Unsupervised Learning** (d.h., ohne Label):  
    - **Word Cloud** pro Korpus  
    - **LDA-Themenmodellierung** für Themenvisualisierung
    - **Word2Vec** auf gesamten Korpus
    3. **Supervised Learning** (d.h., mit Label):  
    - Klassifikation von Sätzen: **Multinomial Naive Bayes** vgl. **Logic Regression**
    - Nutzer*in kann einen Satz eingeben und erhält die **wahrscheinlichste Quelle** (Wissenschaft, Institution oder Medien)
    """)

def show_topic_visualization():
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
    with st.expander(f"Word Cloud des Korpus: {corpus_choice}", expanded=True):
        wc = generate_wordcloud(dictionary, corpus)
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
    Manchmal erscheint dasselbe Wort in mehreren Themen einer einzelnen Gruppe.  
    Das liegt daran, dass dieses Wort im Korpus insgesamt sehr häufig vorkommt und auch in verschiedenen thematischen Zusammenhängen eine wichtige Rolle spielt.  
    LDA: "Latent Dirichlet Allocation".
    """)    

    st.subheader("📐 **Word2Vec:** Semantische Wortbeziehungen entdecken")

    st.markdown("""
    **Word2Vec** ist ein Modell, das Wörter als numerische Vektoren darstellt, 
    basierend auf ihren semantischen Kontexten im Textkorpus.
    Diese Vektoren haben typischerweise eine hohe Dimension (z.B. 100),  
    werden aber mithilfe einer **PCA** (Hauptkomponentenanalyse) auf zwei Dimensionen reduziert,  
    um sie besser visualisieren zu können. 
    Dadurch lassen sich Beziehungen zwischen Begriffen identifizieren 
    und anschaulich im zweidimensionalen Raum darstellen.
    
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
    words_not_found = [w for w in words_checked if w not in model_w2v.wv]

    # Wenn kein Word im Modell
    if not words_found:
        st.error("⚠️ Keine der eingegebenen Wörter ist im Modell vorhanden.")
    else:
        # Einige Wörter sind nicht im Modell
        if words_not_found:
            st.warning(f"⚠️ Folgende Wörter sind **nicht** im Modell enthalten und wurden ignoriert: {', '.join(words_not_found)}")
        
        # Visualisierung für die gefundene Wörter
        fig = plot_word_vectors(model_w2v, words_found)
        if fig:
            st.pyplot(fig)

def show_sentence_classification():
    st.title("🧠 Klassifizierung von Sätzen")
    st.markdown("""
    Gib unten einen beliebigen Satz über Lebenserwartung und Klimmawandel auf Englisch ein und unser Modell sagt dir voraus,
    ob der aus einem **wissenschaftlichen**, **institutionellen** oder **journalistischen** Text stammt.     
    Probiere zum Beispiel mit: "Climate change affects life expectancy"; anschließend füge "in Europe" hinzu.
    """)

    vectorizer, model_lr, model_nb = load_classification_models()

    #try:
    #    check_is_fitted(vectorizer, "idf_")
    #except NotFittedError:
    #    st.error("❌ DerVektorisierer wurde nicht trainiert. Bitte überprüfe die Datei 'vectorizer.pkl'.")
    #    st.stop()
        
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
            st.write(f"• **{label}**: {prob * 100:.1f}%".replace('.', ','))
        
        # Barchart der Wahrscheinlichkeit

        import matplotlib.ticker as mtick

        proba_df = pd.DataFrame({"Klasse": classes, "Wahrscheinlichkeit": probas})
        proba_df = proba_df.sort_values("Wahrscheinlichkeit", ascending=True)

        fig, ax = plt.subplots(figsize=(3, 1.2))
        ax.barh(proba_df["Klasse"], proba_df["Wahrscheinlichkeit"], color="skyblue")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Wahrscheinlichkeit", fontsize=4)
        ax.set_title("Verteilung der Klassifikation", fontsize = 5)
        ax.tick_params(axis='both', labelsize=5)
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}".replace('.', ',')))
        fig.tight_layout(pad=1)
        st.pyplot(fig)

        # Vergleich Multinomial Naive Bayes - Logistic Regression
        st.markdown("## 🤖 Modellauswahl für die Klassifizierung")
        st.markdown(""" Zunächst wurde das Modell **Multinomial Naive Bayes** angewendet.
                    Da die **Accuracy** jedoch mit **67,56%** relativ niedrig ausfiel und 
                    vor allem viele Sätze der Kategorie *academic* zugeordnet wurden (s. *Confusion Matrix*), 
                    wurde angenommen, dass dieses Modell stärker von der **Überrepräsentation 
                    akademischer Sätze** beeinflusst ist.""")
        st.markdown("""Aus diesem Grund wurde zusätzlich **Logistic Regression** getestet.
                    Dieses Modell lieferte nicht nur eine höhere **Accuracy** von **75,35%**, 
                    sondern auch deutlich **ausgewogenere Ergebnisse** in Bezug auf *Precision*, 
                    *Recall* und *F1-Score*, insbesondere bei den Klassen *institutional* und *media*.
                    Daher wurde **Logistic Regression** für die finale Klassifikation bevorzugt.""")
        
        st.markdown("### Confusion Matrix:")

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
            fig, ax = plt.subplots(figsize=(2, 2))
            heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        square= True,
                        xticklabels=labels, yticklabels=labels, 
                        annot_kws={"size": 5}, ax=ax, 
                        cbar_kws={"shrink": 0.5})
            cbar = heatmap.collections[0].colorbar
            cbar.ax.tick_params(labelsize=5)
            
            ax.set_title(title, fontsize=6)
            ax.set_xlabel("Predicted", fontsize=5)
            ax.set_ylabel("Actual", fontsize=5)
            ax.tick_params(axis='both', labelsize=4)
            heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=0, fontsize=3)
            heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=3)

            fig.tight_layout(pad=1)
            st.pyplot(fig)
        

        col1, col2 = st.columns(2)

        with col1:
            plot_confusion_matrix(y_test, nb_preds, labels=classes, title="Naive Bayes")

        with col2:
            plot_confusion_matrix(y_test, lr_preds, labels=classes, title="Logistic Regression")   


        st.markdown("### Classification Reports:")

        
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

        st.markdown("#### Multinomial Naive Bayes")
        st.markdown(f"➡️ **Accuracy**: {accuracy_nb:.2%}")
        st.dataframe(style_report_table(report_nb_df))

        st.markdown("#### Logistic Regression 🏅 ")
        st.markdown(f"➡️ **Accuracy**: {accuracy_lr:.2%}")
        st.dataframe(style_report_table(report_lr_df))
