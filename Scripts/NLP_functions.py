# Funktion um Word Cloud eines ganzen Korpus zu generieren
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim import corpora

def load_dict_and_corpus(path_dict, path_corpus):
    dictionary = corpora.Dictionary.load(path_dict)
    corpus = corpora.MmCorpus(path_corpus)
    return dictionary, corpus

def generate_wordcloud(dictionary, corpus, width=800, height=400):
    word_freq = {}
    for doc in corpus:
        for word_id, freq in doc:
            word = dictionary[word_id]
            word_freq[word] = word_freq.get(word, 0) + freq

    wc = WordCloud(width=width, height=height, background_color="white").generate_from_frequencies(word_freq)
    return wc


# Word Cloud für Thema LDA
def generate_topic_wordcloud(lda_model, topic_id, width=1200, height=920):
    topic_terms = dict(lda_model.show_topic(topic_id, topn=30))
    wc = WordCloud(width=width, height=height, background_color="white").generate_from_frequencies(topic_terms)
    return wc

# Bar Chart für Thema LDA
def generate_topic_bar_chart(lda_model, topic_id, topn=10):
    topic_terms = lda_model.show_topic(topic_id, topn=topn)
    words = [term[0] for term in topic_terms]
    weights = [term[1] for term in topic_terms]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(words[::-1], weights[::-1])
    ax.set_xlabel("Wichtigkeit")
    ax.set_title(f"Thema {topic_id + 1}")
    return fig


# Word2Vec
def plot_word_vectors(model, words):
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    labels = [word for word in words if word in model.wv]
    
    if not word_vectors:
        print("Keine der eingegebenen Wörter ist im Modell vorhanden.")
        return

    pca = PCA(n_components=2)
    coords = pca.fit_transform(word_vectors)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:, 0], coords[:, 1])

    for i, label in enumerate(labels):
        ax.annotate(label, (coords[i, 0], coords[i, 1]))

    ax.set_title("Word2Vec – PCA-Projektion")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True)
    
    return fig  