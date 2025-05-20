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


def show_ca_content():
    # Titel
    st.markdown("<h3 style='margin-bottom: 0;'>🌍 Esperança de vida i canvi climàtic amb PLN</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin-top: 0;'>Núvols de mots, visualització de temes, classificació de frases</h4>", unsafe_allow_html=True)

    # Navigation
    st.sidebar.title("🧭 Navegació")
    option = st.sidebar.radio(
        "Tria una secció:",
        ("📘 Descripció del projecte", 
         "📲 Dades i metodologia", 
         "🗂️ Visualització de temes", 
         "🧠 Classificació de frases", 
         "🚀 Possibles millores",
         "📋 Sobre el projecte")
    )

    if option == "📘 Descripció del projecte":
        show_project_description()
    elif option == "📲 Dades i metodologia":
        show_data_methodology()
    elif option == "🗂️ Visualització de temes":
        show_topic_visualization()
    elif option == "🧠 Classificació de frases":
        show_sentence_classification()
    elif option == "🚀 Possibles millores":
        st.title("🚀 Possibles millores")
        st.markdown("""
        ↗️ **Ampliar el corpus institucional i el periodístic**  
        Així com el corpus acadèmic s'ha pogut fornir amb èxit a través d'una API (Semantic Scholar) i s'ha recopilat un bon nombre d'articles, 
        no s'ha aconseguit el mateix amb els altres dos corpus. 
        L'objectiu és poder extreure igualment de manera automàtica un gran nombre de textos institucionals i periodístics, 
        mitjançant una API o *Web Scraping*.     
        Això augmentaria i equilibraria la base de dades i milloraria la classificació. \n    
        👌**Augmentar la sensibilitat del classificador**         
        Actualment, el model només intenta classificar cada frase dins d’una de les tres fonts predefinides (acadèmica, 
        institucional o periodística). Una millora futura seria incorporar un mecanisme que detecti si la frase introduïda 
        pertany al domini del corpus global. Això permetria oferir **una predicció sobre el grau de versemblança de la frase dins del context analitzat** 
        i alertar quan el seu contingut s’allunya significativament dels textos del corpus original.\n
        🧱 **Millorar la modularitat de l'estructura interna del projecte**         
        La llargada de l’script principal (de_content.py / ca_content.py) es pot reduir i gestionar millor si es separen 
        les funcions en mòduls independents.    
        Així mateix, l’estructura de les dades (models, corpus i diccionaris) pot ser reorganitzada 
        per afavorir una millor mantenibilitat i escalabilitat.      
        """)
    elif option == "📋 Sobre el projecte":
        st.title("📋 Sobre el projecte")
        st.markdown("""
        **Autora:** Ariadna Benet  
        **Projecte realitzat dins el [Data Science Bootcamp – Data Science Institute by Fabian Rappert](https://www.data-science-institute.de)**  
        **Any:** 2025  
        
        📊 Dataset de punt de partida: [Countries Life Expectancy (Kaggle)](https://www.kaggle.com/datasets/amirhosseinmirzaie/countries-life-expectancy)  
        📚 Llibreries: Streamlit, Gensim, scikit-learn, wordcloud, Matplotlib, NumPy, Pandas, Seaborn

        🔧 Aquest projecte empra tècniques de processament del llenguatge natural (PLN), com LDA, Word2Vec i classificadors supervisats.
        """)


def show_project_description():
    st.markdown("<h2 style='margin-bottom: 0;'>📘Descripció del projecte</h2>", unsafe_allow_html=True)
    st.markdown("""
    Benvinguts al nostre projecte de processament del llenguatge natural (PLN)!  
    Aquest treball és part del projecte final **"Esperança de vida: Anàlisis amb intel·ligència de negoci (*Business Intelligence*), 
    aprenentatge automàtic (*Machine Learning*) i processament del llenguatge natural"**, dut a terme en el marc del **Data Science Bootcamp** 
    (formació en ciència de dades) al [Data Science Institute by Fabian Rappert](https://www.data-science-institute.de).

    📊 **Font de dades**  
    El projecte parteix del conjunt de dades 👉 [Countries Life Expectancy](https://www.kaggle.com/datasets/amirhosseinmirzaie/countries-life-expectancy) *(Kaggle)*.
    Les anàlisis amb intel·ligència de negoci i amb aprenentatge automàtic es basen en aquest conjunt de dades.
    Es pregunten quines són les variables més determinants per a l'esperança de vida en tots els continents. I fan previsions sobre com poden evolucionar les xifres 
    en funció de com canviïn els diversos factors.  

    🌎 **Pregunta de recerca**  
    Un cop examinades les dades sobre l'esperança de vida, **ens preguntem si el canvi climàtic afecta o pot afectar l’esperança de vida**. 
    Quins efectes poden tenir sobre la llargada de la vida fenòmens globals com ara l'augment de les temperatures o la contaminació? Són els mateixos en tots els països, rics o pobres? 
    I els efectes més locals del canvi climàtic, com ara les catàstrofes naturals, han deixat evidències ja sobre l'esperança de vida? Els països desenvolupats es podran veure més afectats 
    pel canvi climàtic que per certes malalties, per exemple? 
    
    🔭 **Enfocament**    
    Atesa la dificultat de vincular directament conjunts de dades numèriques dels dos àmbits, optem per un enfocament alternatiu: 
    analitzem textos sobre les dues temàtiques mitjançant tècniques d’aprenentatge automàtic, a fi de respondre les nostres preguntes de recerca.        
     
    🎯 **Objectiu**            
    Ens proposem, doncs, **enllaçar el tema de l'esperança de vida amb el del canvi climàtic** tot analitzant **textos en anglès** sobre els dos temes, provinents de **tres fonts diferents**:
    
    - 👩‍🏫 Publicacions acadèmiques  
    - 🏛️ Informes institucionals  
    - 📰 Articles periodístics  
           
    🗂️ Examinem els temes més freqüents en textos procedents de les tres fonts.       
    🧠 Entrenem un classificador de frases que prediu de quina d’aquestes tres fonts prové una frase determinada.
                
    ⚙️ **Metodologia**     
    Mitjançant l'**anàlisi de temes** (*Topic Modeling*), **representacions semàntiques de paraules** (*Word Embeddings*) i **mètodes de classificació** (amb *Multinomial Naive Bayes* i regressió logística), 
    obtenim una visió sobre la representació lingüística d'aquestes temàtiques globals.   
    A manca de dades numèriques que enllacin els dos àmbits, **convertim textos i paraules en vectors i xifres**.
                
    """, unsafe_allow_html=True)

def show_data_methodology():
    st.markdown("<h2 style='margin-bottom: 0;'>📲 Dades i metodologia</h2>", unsafe_allow_html=True)
    # Daten
    st.subheader("🔍 Dades recollides")
    st.markdown("""
    Els corpus s'han recollit i netejat mitjançant API i manualment. En total analitzem:

    - 432 resums acadèmics (API de Semantic Scholar)
    - 22 informes institucionals (OMS, ONU, Banc Mundial)
    - 29 articles periodístics (BBC, Euronews, Reuters, The Conversation, entre d'altres)
                
    Mots clau per a la cerca: "life expectancy" AND "climate change"; "health" AND "climate change"
    
    Objectiu d'optimització: 
    Per aconseguir un millor equilibri entre les tres fonts, 
    caldria ampliar els corpus institucional i periodístic —idealment mitjançant *scraping* automatitzat amb APIs 
    (vegeu 🚀 Possibles millores).            
 
    En total treballem amb més de **154.000 paraules**.
    """)
 
    # Daten
    labels = ["Acadèmia", "Institucions", "Mitjans de comunicació"]
    sizes = [96909, 29716, 27720]
    colors = ["#66b3ff", "#99ff99", "#ffcc99"]
    total = sum(sizes)

    # Bargraph
    fig, ax = plt.subplots(figsize=(4, 1.8))  

    bars = ax.barh(labels, sizes, color=colors)
    ax.set_xlabel("Nombre de mots", fontsize=8)
    ax.set_title("Distribució dels corpus segons la font", fontsize=8)

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
            return f"{pct:.1f}%\n({val:,} mots)".replace(',', 'X').replace('.', ',').replace('X', '.')
        return my_autopct



    # Methoden
    st.subheader("⚙️ Mètodes i procediment")
    st.markdown("""
    Per al desenvolupament del projecte, s'han dut a terme aquests passos:

    1. **Preprocessament**: tot el text a minúscules, eliminació de caràcters especials i separacions de línia.
    A més, específicament per a cada part: 
    
        
    | Mòdul              | Tokenització      | Unitat d'anàlisi | *Bigrams* | *Stopwords*                              | Etiqueta |
    |--------------------|-------------------|------------------|-----------|------------------------------------------|----------|
    | **Núvols de mots**     | ✅        | Mot             | ✅      | Anglès, mots clau, específics del corpus | ❌     |
    | **LDA**            | ✅                | Mot             | ✅      | Anglès,  mots clau, específics del corpus | ❌     |
    | **Word2Vec**       | ✅                | Mot             | ✅      | Anglès, específics del corpus                | ❌     |
    | **Classificació** | ✅                | Frase             | ❌      | cap                                     | ✅     |

    """)           
    with st.expander("✂️ Què és la *tokenització*?"):
        st.write("""**La tokenització** és el primer pas en l’anàlisi automatitzada del llenguatge natural (PLN).   
        \nConsisteix a dividir un text en unitats més petites anomenades *tokens*, com ara paraules o frases.   
        \nAquests tokens serveixen com a base per a l’anàlisi posterior, com la detecció de temes o la classificació.""")

    with st.expander("👯‍♀️ Què és un *bigram*?"):
        st.write("""Un **bigram** és una parella de paraules consecutives que sovint apareixen juntes en un text
        i que formen una unitat de significat.
        \n**Exemple:** la parella *"air pollution"* és un bigram perquè aquestes dues paraules sovint apareixen juntes
        i signifiquen *contaminació de l’aire*.""")

    with st.expander("🫸 Què són els *stopwords*?"):
        st.write("""Els **stopwords** (o **mots buits**) són paraules molt freqüents com ara *i*, *el*, *és* o *de*,
        que habitualment no aporten gaire informació i es descarten en les anàlisis.
        \n🔹 **Anglès:** Per a cada llengua s'apliquen llistes de mots buits específiques. Es tracta sobretot de 
                 mots funcionals, com ara en anglès, per exemple, *the*, *is*, *and*, etc.  
        🔹 **Mots clau:** Termes massa dominants com ara *health* o *climate change* també s'eliminen, per evitar biaixos.  
        🔹 **Específics del corpus:** s'exclouen paraules molt freqüents dins d’un corpus concret però amb poc valor analític —com *study*, *analysis*, o *conclusions* en textos acadèmics.
        """)
    
    st.markdown("""
    
    2. **Aprenentatge no supervisat** (és a dir, sense etiquetes):  
    - **Núvols de mots** per a cada corpus  
    - **Modelatge de temes amb LDA** per visualitzar temes
    - **Word2Vec** aplicat a tot el corpus
    3. **Aprenentatge supervisat** (és a dir, amb etiquetes):  
    - Classificació de frases: amb els models **Multinomial Naive Bayes** i **Regressió logística**
    - L'usuari pot introduir una frase i rebrà la font més probable (corpus acadèmic, institucional o periodístic)
    """)

def show_topic_visualization():
    st.title("🗂️ Visualització de temes")
    st.markdown("""
    Aquí es mostren els temes més rellevants de cada corpus mitjançant: 
    
    - 🔠 **Núvols de mots**  
    - 📊 **Modelatge de temes amb LDA** 
    - 📐 **Word2Vec**
    """)

    st.subheader("🔠 **Núvols de mots**")
 
    # Auswahl des Korpuses

    corpus_labels = {
        "Wissenschaft": "Acadèmic",
        "Institutionen": "Institucional",
        "Medien": "Periodístic"
    }

    corpus_choice_label = st.selectbox(
        "Tria un corpus:",
        list(corpus_labels.values()),
        key="lda_corpus"
    )

    label_to_key = {v: k for k, v in corpus_labels.items()}
    corpus_choice = label_to_key[corpus_choice_label]

    # Load Wörterbuch und Korpus
    path_dict = corpus_paths[corpus_choice]["dict"]
    path_corpus = corpus_paths[corpus_choice]["corpus"]
    dictionary, corpus = load_dict_and_corpus(path_dict, path_corpus)

    # LDA model corresponent
    lda_model = lda_models[corpus_choice]

    # Word Cloud generieren und zeigen
    with st.expander(f"Núvol de mots del corpus: {corpus_choice_label}", expanded=True):
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
            labels.append((i, f"Tema {i+1} – {main_word}"))
        return labels
    
    # Themenliste aufbauen
    topic_labels = get_topic_labels(lda_model)
    label_to_id = {label: idx for idx, label in topic_labels}

    st.subheader("📊 **Modelatge de temes amb LDA**" )

    # Thema auswählen
    topic_label = st.selectbox(f"5 Top-Temes del corpus {corpus_choice_label}", list(label_to_id.keys()), key="lda_topic")
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
    Aquests temes han estat extrets del corpus mitjançant un model LDA.  
    Cada tema està format per paraules que sovint apareixen juntes en un mateix context.  
    De vegades, una mateixa paraula pot aparèixer en diversos temes d'un grup. Això passa perquè és una paraula molt freqüent 
    en el corpus i juga un paper rellevant en diferents àmbits temàtics.  
    LDA: *Latent Dirichlet Allocation*.
    """)
   

    st.subheader("📐 **Word2Vec:** Descobrint relacions semàntiques dels mots")

    st.markdown("""
    **Word2Vec** és un model que representa les paraules com a vectors numèrics, basant-se en els contextos semàntics del mot dins del corpus textual.  
    Aquests vectors solen tenir una dimensionalitat alta (p. ex., 100), però es redueixen a dues dimensions 
    mitjançant una **PCA** (anàlisi de components principals) per tal de poder-los visualitzar millor.  
    Aquesta representació permet identificar relacions entre termes i mostrar-les de manera clara en un espai bidimensional.

    📍 En el següent gràfic es pot veure una projecció PCA d’aquests vectors.  
    Com més a prop es troben dos punts, més semblants han estat interpretats pel model.
    """)


    # Modell laden
    @st.cache_resource
    def load_model():
        return Word2Vec.load("models/word2vec_global.model")
    
    model_w2v = load_model()

    # 🌐 PCA-Visualisierung

    input_words = st.text_input(
        "Mots que vols visualitzar (separats per espai):",
        "climate_change life_expectancy health air_pollution policy"
    )

    words_list = input_words.strip().split()
    words_checked = [w.lower().strip() for w in input_words.strip().split()]
    words_found = [w for w in words_checked if w in model_w2v.wv]
    words_not_found = [w for w in words_checked if w not in model_w2v.wv]

    # Wenn kein Word im Modell
    if not words_found:
        st.error("⚠️ Cap dels mots es troben dins el model.")
    else:
        # Einige Wörter sind nicht im Modell
        if words_not_found:
            st.warning(f"⚠️ Els següents mots **no** es troben dins el model i han estat ignorats: {', '.join(words_not_found)}")
        
        # Visualisierung für die gefundene Wörter
        fig = plot_word_vectors(model_w2v, words_found)
        if fig:
            st.pyplot(fig)

def show_sentence_classification():
    st.title("🧠 Classificació de frases")
    st.markdown("""Escriu una frase en anglès sobre l’esperança de vida i el canvi climàtic, i el nostre model et donarà 
                la predicció de la font original més probable —és a dir, si la frase és més probable de ser trobada 
                en un text acadèmic, institucional o periodístic.    
                Prova per exemple amb: "Climate change affects life expectancy"; després afegeix "in Europe".
                """)

    vectorizer, model_lr, model_nb = load_classification_models()

    try:
        check_is_fitted(vectorizer, "idf_")
    except NotFittedError:
        st.error("❌ El vectoritzador no ha estat entrenat. Sisplau, comprova l'arxiu 'vectorizer.pkl'.")
        st.stop()
        
    # Benutzereingabe
    user_input = st.text_input("✏️ Escriu una frase:")

    label_translation = {
        "academic": "acadèmic",
        "institutional": "institucional",
        "media": "periodístic"
    }

    if user_input:
        # In Vektoren umwandeln und vorhersagen
        X_input = vectorizer.transform([user_input])
        pred = model_lr.predict(X_input)[0]
        probas = model_lr.predict_proba(X_input)[0]
        classes = model_lr.classes_

        pred_label = label_translation.get(pred, pred)
        st.success(f"Font més probable: **corpus {pred_label}**")

        # Wahrscheinlichkeiten anzeigen
        st.markdown("### 🎲 Probabilitats:")
        for label, prob in zip(classes, probas):
            translated_label = label_translation.get(label, label)
            st.write(f"• Corpus **{translated_label}**: {prob * 100:.1f}%".replace('.', ','))
        
        # Barchart der Wahrscheinlichkeit

        import matplotlib.ticker as mtick

        translated_classes = [label_translation.get(label, label) for label in classes]
        proba_df = pd.DataFrame({"Classe": translated_classes, "Probabilitat": probas})
        proba_df = proba_df.sort_values("Probabilitat", ascending=True)

        fig, ax = plt.subplots(figsize=(3, 1.2))
        ax.barh(proba_df["Classe"], proba_df["Probabilitat"], color="skyblue")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probabilitat", fontsize=4)
        ax.set_title("Distribució de la classificació", fontsize = 5)
        ax.tick_params(axis='both', labelsize=5)
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}".replace('.', ',')))
        fig.tight_layout(pad=1)
        st.pyplot(fig)

        # Vergleich Multinomial Naive Bayes - Logistic Regression
        st.markdown("## 🤖 Elecció del model per a la classificació")
            
        st.markdown("""En primer lloc, es va aplicar el model **Multinomial Naive Bayes**.  
                        Però com que com que la **precisió** era relativament baixa (**67,56%**),  
                        i sobretot perquè molts enunciats van ser classificats com a *acadèmics* (vegeu la *matriu de confusió*),  
                        es va considerar que aquest model estava fortament influït per la **sobrerrepresentació de frases acadèmiques**.
                        """)
        st.markdown("""Per això a continuació es va provar la **regressió logística**.  
                        Aquest model no només ofereix una **precisió** més alta (**75,35%**),  
                        sinó també uns resultats **molt més equilibrats** pel que fa a *Precision*,  
                        *Recall* i *F1-score*, especialment en les classes *institucional* i *periodística*.  
                        Finalment, doncs, es va optar per la **regressió logística** com a model per a la classificació.
                        """)
        
        st.markdown("### Matriu de confusió:")

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
            plot_confusion_matrix(y_test, lr_preds, labels=classes, title="Regressió logística")   


        st.markdown("### Informes de classificació:")

        
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
        st.markdown(f"➡️ **Precisió**: {accuracy_nb:.2%}")
        st.dataframe(style_report_table(report_nb_df))

        st.markdown("#### Logistic Regression 🏅 ")
        st.markdown(f"➡️ **Precisió**: {accuracy_lr:.2%}")
        st.dataframe(style_report_table(report_lr_df))
