import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Détecteur de Minerais Nikki-Parakou", page_icon="⛏️")

st.title("⛏️ Détecteur de Minerais - Nikki / Parakou")
st.write("Application IA pour détecter les minerais dans le sous-sol avec la géophysique.")

# --- Données d'entraînement (exemple réaliste) ---
data = {
    'resistivite': [600, 550, 620, 50, 45, 60, 180, 200, 160, 120, 110, 130, 300, 400, 90],
    'magnetisme': [50, 60, 40, 500, 550, 480, 20, 25, 30, 40, 35, 45, 80, 70, 60],
    'arsenic': [12, 15, 11, 1, 0.5, 1.2, 0.2, 0.3, 0.1, 0.5, 0.4, 0.6, 2, 1, 0.8],
    'fer': [3, 4, 2, 45, 50, 42, 5, 6, 4, 8, 7, 9, 10, 12, 15],
    'minerai': ['Or', 'Or', 'Or', 'Fer', 'Fer', 'Fer', 'Phosphate', 'Phosphate', 'Phosphate', 'Rien', 'Rien', 'Rien', 'Rien', 'Rien', 'Rien']
}
df = pd.DataFrame(data)
X = df[['resistivite', 'magnetisme', 'arsenic', 'fer']]
y = df['minerai']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# --- Interface ---
st.sidebar.header("Mesures du terrain")
res = st.sidebar.slider("Résistivité (Ohm.m)", 0, 1000, 150)
mag = st.sidebar.slider("Magnétisme (nT)", 0, 1000, 50)
ars = st.sidebar.slider("Arsenic (ppm)", 0.0, 20.0, 1.0)
fer_val = st.sidebar.slider("Teneur en Fer (%)", 0, 60, 5)

if st.button("ANALYSER LE TERRAIN"):
    prediction = model.predict([[res, mag, ars, fer_val]])[0]
    proba = model.predict_proba([[res, mag, ars, fer_val]]).max() * 100
    
    if prediction == 'Or':
        st.success(f"🟡 OR DÉTECTÉ ! Confiance : {proba:.1f}%")
        st.write("Recommandation : Faire un forage de confirmation à Nikki.")
    elif prediction == 'Fer':
        st.warning(f"🔴 FER DÉTECTÉ ! Confiance : {proba:.1f}%")
        st.write("Recommandation : Zone favorable pour le fer.")
    elif prediction == 'Phosphate':
        st.info(f"⚪ PHOSPHATE DÉTECTÉ ! Confiance : {proba:.1f}%")
        st.write("Recommandation : Zone favorable pour le phosphate.")
    else:
        st.error(f"⚫ AUCUN MINERAI. Confiance : {proba:.1f}%")
        st.write("Recommandation : Changer de zone.")

st.markdown("---")
st.caption("Projet de Dossou Antoine - Détection par IA - Nikki-Parakou")
