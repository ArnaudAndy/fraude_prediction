import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration de la page ---
st.set_page_config(
    page_title="Telecom Fraud Shield", 
    page_icon="🛡️", 
    layout="wide"
)

# --- Fonctions de chargement ---
@st.cache_resource
def load_advanced_model():
    """Charge le modèle entraîné sur tous les paramètres."""
    model_path = 'telecom_fraud_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    return None

@st.cache_data
def get_data():
    """Charge les données pour la page Analytics."""
    file_path = 'ICalls dataset.xlsx'
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        return df
    return None

# --- Chargement initial ---
model = load_advanced_model()

# --- Barre latérale de navigation ---
st.sidebar.title("🛡️ Fraud Guard")
page = st.sidebar.radio("Navigation", ["🏠 Home", "🤖 Détection (Model)", "📊 Analytics"])

# ==========================================
# PAGE : HOME
# ==========================================
if page == "🏠 Home":
    st.title("🛡️ Système de Détection de Fraude Internationale")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("À propos de l'application")
        st.write("""
        Cette application utilise l'intelligence artificielle pour sécuriser les réseaux télécom. 
        Elle analyse les appels internationaux en temps réel pour détecter des comportements 
        atypiques pouvant correspondre à des fraudes (Wangiri, SIM Box, détournement de ligne).
        """)
        
        st.subheader("Comment ça marche ?")
        st.info("""
        1. **Collecte** : Nous récupérons les données techniques de l'appel (Switch IDs, Pays, Durée).
        2. **Analyse IA** : L'algorithme **Isolation Forest** compare cet appel à des millions d'appels normaux.
        3. **Diagnostic** : Si l'appel est trop différent de la norme, il est marqué comme 'Anomalie'.
        """)

    with col2:
        st.image("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&q=80&w=400", caption="Sécurité des Réseaux")

    st.header("Le Modèle d'Apprentissage")
    st.write("""
    Nous utilisons **Isolation Forest**, un modèle d'apprentissage non-supervisé. Contrairement aux modèles 
    classiques, il n'a pas besoin de savoir à l'avance ce qu'est une fraude ; il apprend par lui-même 
    que ce qui est rare est suspect.
    """)
    

# ==========================================
# PAGE : MODEL
# ==========================================
elif page == "🤖 Détection (Model)":
    st.title("🤖 Analyse de Risque en Temps Réel")
    st.markdown("---")

    if model is None:
        st.error("⚠️ Le fichier 'telecom_fraud_model.pkl' est introuvable. Relancez votre Notebook.")
        st.stop()

    with st.form("fraud_form"):
        st.subheader("📊 Paramètres de Consommation")
        c1, c2, c3 = st.columns(3)
        with c1:
            duration = st.number_input("Durée de l'appel (sec)", min_value=0, value=60)
            answered = st.selectbox("Répondu ?", options=[1, 0], format_func=lambda x: "Oui" if x==1 else "Non")
        with c2:
            cost = st.number_input("Coût de l'appel ($)", min_value=0.0, format="%.2f", value=0.50)
        with c3:
            revenue = st.number_input("Revenu généré ($)", min_value=0.0, format="%.2f", value=0.60)

        st.subheader("⚙️ Identifiants Techniques")
        c4, c5 = st.columns(2)
        with c4:
            in_switch = st.number_input("IN_SWITCH_ID", value=3)
        with c5:
            hour = st.slider("Heure de l'appel", 0, 23, 14)

        submit = st.form_submit_button("🛡️ LANCER L'ANALYSE")

    if submit:
        # 1. Calcul des variables dérivées (Feature Engineering)
        cost_per_sec = cost / (duration + 1e-6)
        wangiri_score = 1 if (answered == 0 and duration <= 5) else 0
        cost_revenue_diff = cost - revenue

        # 2. Vecteur de 8 variables (Ordre exact du Notebook)
        input_data = np.array([[
            duration, answered, cost, revenue, 
            cost_per_sec, wangiri_score, cost_revenue_diff, in_switch
        ]])
        
        # 3. Prédiction
        prediction = model.predict(input_data)[0]
        score = model.decision_function(input_data)[0]

        # --- SÉCURITÉ ANTI-ABSURDE (Correction demandée) ---
        # Si le coût est massivement supérieur au revenu (ex: > 100$ d'écart),
        # on force l'anomalie même si l'IA hésite.
        if cost_revenue_diff > 100:
            prediction = -1

        st.subheader("Rapport de Diagnostic")
        if prediction == -1:
            st.error(f"🚨 ALERTE : COMPORTEMENT FRAUDULEUX DÉTECTÉ")
            st.metric("Score d'Anomalie", f"{score:.4f}", delta="CRITIQUE", delta_color="inverse")
            st.warning(f"Alerte financière : L'écart coût/revenu est de {cost_revenue_diff:.2f} $")
        else:
            st.success(f"✅ APPEL CONFORME")
            st.metric("Score de Confiance", f"{score:.4f}")

# ==========================================
# PAGE : ANALYTICS
# ==========================================
elif page == "📊 Analytics":
    st.title("📊 Analyses des Flux Internationaux")
    st.write("Exploration visuelle des données d'appels pour identifier les tendances.")
    st.markdown("---")

    df = get_data()
    
    if df is not None:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Répartition par Pays d'Origine")
            top_countries = df['ORIG_COUNTRY_ID'].value_counts().head(10)
            fig1, ax1 = plt.subplots()
            top_countries.plot(kind='bar', ax=ax1, color='skyblue')
            plt.xticks(rotation=45)
            st.pyplot(fig1)
            

        with col_b:
            st.subheader("Relation Coût vs Revenu")
            fig2, ax2 = plt.subplots()
            sns.scatterplot(data=df.sample(min(2000, len(df))), x='COST', y='REVENUE', alpha=0.5, ax=ax2)
            st.pyplot(fig2)
            st.write("Une déviation importante de la ligne diagonale indique une anomalie de facturation.")
            

        st.subheader("Analyse des Switchs (IN vs OUT)")
        st.write("Certains switchs peuvent être plus vulnérables ou utilisés pour des types de trafic spécifiques.")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=df, x='IN_SWITCH_ID', y='CALL_DURATION_SEC', ax=ax3)
        st.pyplot(fig3)
        
    else:
        st.warning("Veuillez placer le fichier 'ICalls dataset.xlsx ' dans le répertoire pour voir les graphiques.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("Développé pour la détection de fraudes Télécom v1.0")