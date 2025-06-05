st.markdown("""
### 🔍 Comparació semàntica entre paraules

Aquesta secció utilitza vectors semàntics **preentrenats** mitjançant el model `en_core_web_md` de [spaCy](https://spacy.io/).  
Per tant, **no es basa en el corpus específic d’aquest projecte**, sinó en coneixement lingüístic general obtingut a partir de textos com Wikipedia, notícies i llibres.

Això permet identificar com de properes són semànticament paraules com *climate*, *pollution* o *inequality* segons el context general de l'anglès.
""")


import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nlp = spacy.load("en_core_web_md")

# Entrada d’usuari
words = st.text_input("Introdueix paraules separades per comes", "climate, pollution, inequality")

word_list = [w.strip() for w in words.split(",") if w.strip() != ""]
vectors = [nlp(w).vector for w in word_list]
sim_matrix = cosine_similarity(vectors)

# Mostra taula de similituds
st.write("### 📊 Matriu de similitud")
import pandas as pd
df_sim = pd.DataFrame(sim_matrix, index=word_list, columns=word_list)
st.dataframe(df_sim.style.format("{:.2f}"))
