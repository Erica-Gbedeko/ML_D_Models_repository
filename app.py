# ÉTAPE 1 : IMPORTER LES BIBLIOTHÈQUES
# =============================================================================
import re
from pathlib import Path
 
import joblib                 # charger les modèles sauvegardés (.joblib)
import numpy as np            # calcul numérique
import pandas as pd           # manipulation des tableaux de données
import streamlit as st        # création de l'interface web
 
 
# =============================================================================
# ÉTAPE 2 : CONFIGURER LA PAGE ET LE STYLE
# (st.set_page_config doit être la 1re commande Streamlit du fichier)
# =============================================================================
APP_NAME = "Erica's ML App"
 
st.set_page_config(
    page_title=f"{APP_NAME} | Applications ML",
    page_icon="🤖",
    layout="wide",
)
 
# Style visuel (CSS) : bandeau d'en-tête, cartes de résultat, boutons.
# Les couleurs sont semi-transparentes : elles s'adaptent au mode clair/sombre.
st.markdown(
    """
    <style>
    /* Bandeau d'en-tête */
    .hero {
        background: linear-gradient(120deg, black 0%, #0EA5E9 100%);
        padding: 1.6rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.2rem;
    }
    .hero h1 { color: #4F46E5; margin: 0; font-size: 2rem; }
    .hero p  { color: #FFFFF4; margin: 0.3rem 0 0 0; font-size: 1.05rem; }
 
    /* Carte de résultat */
    .result-card {
        background: rgba(79, 70, 229, 0.10);
        border-left: 6px solid black;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        margin: 1rem 0 0.8rem 0;
    }
    .result-card .label { font-size: 0.95rem; opacity: 0.75; margin: 0; }
    .result-card .value { font-size: 2.1rem; font-weight: 700; margin: 0.1rem 0 0 0; }
 
    /* Boutons de la barre latérale sur toute la largeur */
    [data-testid="stSidebar"] .stButton > button { width: 100%; border-radius: 10px; }
 
    /* Boutons d'action principaux */
    .stForm button, .stDownloadButton button { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)
 
 
# =============================================================================
# ÉTAPE 3 : DÉCRIRE LES 4 APPLICATIONS (dictionnaire PAGES)
# Chaque modèle est décrit une seule fois ici ; le reste du code est générique.
#   - "features" : champs du formulaire, DANS L'ORDRE des colonnes du notebook
#   - "enc"      : position de la colonne dans encoders.joblib / uniques.joblib
#   - "scalers"  : scaler(s) à appliquer avant la prédiction
# =============================================================================
def _split_adresse(df):
    """Car data : le notebook sépare « Quartier,Ville » depuis la colonne Adresse."""
    if "Adresse" in df.columns:
        parts = df["Adresse"].astype(str).str.split(",", n=1, expand=True)
        df["Quartier"] = parts[0].str.strip()
        df["Ville"] = parts[1].str.strip() if parts.shape[1] > 1 else ""
        df = df.drop(columns="Adresse")
    return df
 
 
PAGES = {
    # ------------------------------------------------------------------ #
    "Car data": {
        "icon": "",
        "title": "Car data : voitures d'occasion ou venant (Dakar)",
        "description": ("Ce modèle (Random Forest) prédit si une voiture est "
                        "**d'occasion** ou **venant** à partir de sa marque, de son année, "
                        "de sa boîte de vitesse, de son prix et de son quartier."),
        "folder": "car_data",
        "task": "classification",
        "scalers": ["scaler.joblib"],            # MinMaxScaler
        "target": "Venant_Occasion",
        "target_label": "Catégorie prédite",
        "target_enc": 2,                         # encoders[2] = Venant_Occasion
        "prep": _split_adresse,
        "csv_note": "Une colonne « Adresse » au format « Quartier,Ville » est aussi acceptée.",
        "features": [
            {"name": "Marque", "label": "Marque", "kind": "cat", "enc": 0, "default": "Ford"},
            {"name": "Année", "label": "Année", "kind": "num", "int": True,
             "min": 1990, "max": 2025, "default": 2015, "step": 1},
            {"name": "Boite_Vitesse", "label": "Boîte de vitesse", "kind": "cat", "enc": 1},
            {"name": "Prix", "label": "Prix (FCFA)", "kind": "num", "int": True,
             "min": 0, "default": 5_000_000, "step": 100_000},
            {"name": "Quartier", "label": "Quartier", "kind": "cat", "enc": 3,
             "default": "Almadies"},
        ],
    },
    # ------------------------------------------------------------------ #
    "Car dataset": {
        "icon": "",
        "title": "Car dataset : prix de vente d'une voiture",
        "description": ("Ce modèle (Random Forest) prédit le **prix de vente** d'une voiture "
                        "à partir de son kilométrage, de son prix actuel, de son carburant, "
                        "du type de vendeur, de sa transmission et de son âge."),
        "folder": "car_dataset",
        "task": "regression",
        "scalers": ["scaler.joblib"],            # RobustScaler
        "target": "Selling_Price",
        "target_label": "Prix de vente estimé",
        "unit": "$",
        "features": [
            {"name": "Kms_Driven", "label": "Kilométrage (km)", "kind": "num", "int": True,
             "min": 0, "default": 50_000, "step": 1000},
            {"name": "Present_Price", "label": "Prix actuel (neuf)", "kind": "num",
             "min": 0.0, "default": 10.0, "step": 0.5},
            {"name": "Fuel_Type", "label": "Carburant", "kind": "cat", "enc": 0,
             "default": "Petrol"},
            {"name": "Seller_Type", "label": "Type de vendeur", "kind": "cat", "enc": 1,
             "default": "Dealer"},
            {"name": "Transmission", "label": "Transmission", "kind": "cat", "enc": 2},
            {"name": "Age", "label": "Âge (années)", "kind": "num", "int": True,
             "min": 0, "max": 40, "default": 5, "step": 1},
        ],
    },
    # ------------------------------------------------------------------ #
    "Telephone": {
        "icon": "",
        "title": "Telephone : état d'un téléphone",
        "description": ("Ce modèle (Random Forest) prédit l'**état** d'un téléphone "
                        "(neuf, d'occasion, reconditionné, venant) à partir de son prix, "
                        "de son adresse, de sa marque, de la taille d'écran, de la RAM "
                        "et du stockage."),
        "folder": "telephone",
        "task": "classification",
        "scalers": ["scaler_robust.joblib", "scaler_minmax.joblib"],  # appliqués dans cet ordre
        "target": "etat",
        "target_label": "État prédit",
        "target_enc": 2,                         # encoders[2] = etat
        "features": [
            {"name": "prix", "label": "Prix (FCFA)", "kind": "num", "int": True,
             "min": 0, "default": 300_000, "step": 10_000},
            {"name": "adresse", "label": "Adresse", "kind": "cat", "enc": 0,
             "default": "Plateau"},
            {"name": "marque", "label": "Marque", "kind": "cat", "enc": 1,
             "default": "Samsung"},
            {"name": "dim_ecr", "label": "Taille de l'écran (pouces)", "kind": "num",
             "min": 0.0, "max": 80.0, "default": 6.0, "step": 0.1},
            {"name": "ram", "label": "RAM (Go)", "kind": "num", "int": True,
             "min": 1, "max": 512, "default": 8, "step": 1},
            {"name": "stockage", "label": "Stockage (Go)", "kind": "num", "int": True,
             "min": 1, "max": 5120, "default": 128, "step": 1},
        ],
    },
    # ------------------------------------------------------------------ #
    "Taxi trips": {
        "icon": "",
        "title": "Taxi trips : prix d'une course",
        "description": ("Ce modèle (Random Forest) prédit le **prix d'une course de taxi** à "
                        "partir de la distance, du moment de la journée, du jour, du nombre "
                        "de passagers, du trafic, de la météo, des tarifs et de la durée."),
        "folder": "taxi_trip",
        "task": "regression",
        "scalers": ["scaler.joblib"],            # RobustScaler
        "target": "Trip_Price",
        "target_label": "Prix de la course",
        "unit": "$",
        "features": [
            {"name": "Trip_Distance_km", "label": "Distance (km)", "kind": "num",
             "min": 0.0, "default": 25.0, "step": 0.5},
            {"name": "Time_of_Day", "label": "Moment de la journée", "kind": "cat", "enc": 0},
            {"name": "Day_of_Week", "label": "Jour de la semaine", "kind": "cat", "enc": 1},
            {"name": "Passenger_Count", "label": "Nombre de passagers", "kind": "num",
             "int": True, "min": 1, "max": 4, "default": 2, "step": 1},
            {"name": "Traffic_Conditions", "label": "Trafic", "kind": "cat", "enc": 2},
            {"name": "Weather", "label": "Météo", "kind": "cat", "enc": 3},
            {"name": "Base_Fare", "label": "Tarif de base", "kind": "num",
             "min": 0.0, "default": 3.5, "step": 0.1},
            {"name": "Per_Km_Rate", "label": "Tarif par km", "kind": "num",
             "min": 0.0, "default": 1.2, "step": 0.05},
            {"name": "Per_Minute_Rate", "label": "Tarif par minute", "kind": "num",
             "min": 0.0, "default": 0.3, "step": 0.01},
            {"name": "Trip_Duration_Minutes", "label": "Durée (minutes)", "kind": "num",
             "min": 0.0, "default": 60.0, "step": 1.0},
        ],
    },
}
 
 
# =============================================================================
# ÉTAPE 4 : CHARGER LES OBJETS SAUVEGARDÉS DANS LES NOTEBOOKS
# Pour chaque modèle on charge 4 éléments avec joblib :
#   le modèle, les encodeurs, les valeurs uniques et le(s) scaler(s).
# @st.cache_resource : le chargement se fait UNE seule fois (pas à chaque clic).
# =============================================================================
APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"
 
 
def _norm(name: str) -> str:
    """Normalise un nom de dossier (sans majuscules, espaces, « _ » ni « - »)."""
    return re.sub(r"[\s_\-]", "", name.lower())
 
 
def find_model_dir(folder: str) -> Path:
    """Retrouve le dossier du modèle dans models/ (ou à la racine du projet)."""
    key = _norm(folder)
    for base in (MODELS_DIR, APP_DIR):
        if base.is_dir():
            for p in base.iterdir():
                if p.is_dir() and _norm(p.name) == key:
                    return p
    return MODELS_DIR / folder
 
 
@st.cache_resource
def load_artifacts(folder: str, scaler_files: tuple):
    """Charge le modèle, les encodeurs, les valeurs uniques et les scalers."""
    d = find_model_dir(folder)
    return {
        "model": joblib.load(d / "rf_model.joblib"),
        "encoders": joblib.load(d / "encoders.joblib"),
        "uniques": joblib.load(d / "uniques.joblib"),
        "scalers": [joblib.load(d / f) for f in scaler_files],  # appliqués dans l'ordre
    }
 
 
def check_artifacts(cfg, art):
    """Vérifie que les fichiers du dossier correspondent bien à cette page."""
    cats = [f["enc"] for f in cfg["features"] if f["kind"] == "cat"]
    if cfg["task"] == "classification":
        cats.append(cfg["target_enc"])
    need = max(cats) + 1 if cats else 0
    if len(art["uniques"]) < need or len(art["encoders"]) < need:
        return ("Ce dossier ne contient pas le bon modèle pour cette page : "
                "relancez le bon notebook puis réexportez les fichiers .joblib.")
    for i in set(cats):
        if art["encoders"][i].classes_.dtype.kind in "iufb":
            return ("Les encodeurs contiennent des nombres au lieu des noms : "
                    "relancez tout le notebook dans l'ordre puis réexportez les .joblib.")
    n = getattr(art["model"], "n_features_in_", None)
    if n is not None and n != len(cfg["features"]):
        return f"Le modèle attend {n} variables, mais cette page en utilise {len(cfg['features'])}."
    return None
 
 
# =============================================================================
# ÉTAPE 5 : FONCTION DE PRÉDICTION
# On refait EXACTEMENT les mêmes transformations que dans le notebook :
#   1) encoder les variables catégorielles   (encoders)
#   2) normaliser les données                (scaler)
#   3) prédire                               (model.predict)
# Cette fonction sert aux deux onglets (simple et multiple).
# =============================================================================
def predire(cfg, art, df):
    """Retourne (prédictions, probabilités). Les probabilités = None en régression."""
    names = [f["name"] for f in cfg["features"]]
    X = df[names].copy()
 
    # 1) Encodage des colonnes catégorielles avec les encodeurs du notebook
    for f in cfg["features"]:
        if f["kind"] == "cat":
            enc = art["encoders"][f["enc"]]
            X[f["name"]] = enc.transform(X[f["name"]].astype(str).str.strip())
    X = X.apply(pd.to_numeric).to_numpy(dtype=float)
 
    # 2) Normalisation avec le(s) scaler(s) du notebook
    for scaler in art["scalers"]:
        X = scaler.transform(X)
 
    # 3) Prédiction avec le modèle Random Forest
    model = art["model"]
    y = model.predict(X)
 
    # Régression : on renvoie directement le prix (arrondi)
    if cfg["task"] == "regression":
        return np.round(y.astype(float), 2), None
 
    # Classification : on retrouve le NOM de la classe + les probabilités
    target_enc = art["encoders"][cfg["target_enc"]]
    labels = target_enc.inverse_transform(y.astype(int))
    proba = pd.DataFrame(
        model.predict_proba(X),
        columns=target_enc.inverse_transform(model.classes_.astype(int)),
    )
    return labels, proba
 
 
def carte_resultat(titre, valeur):
    """Affiche un résultat dans une jolie carte."""
    st.markdown(
        f'<div class="result-card"><p class="label">{titre}</p>'
        f'<p class="value">{valeur}</p></div>',
        unsafe_allow_html=True,
    )
 
 
# =============================================================================
# ÉTAPE 6 : ONGLET 1 : PRÉDICTION SIMPLE (un formulaire)
# Les listes déroulantes sont remplies avec les vraies valeurs des encodeurs :
# l'utilisateur ne peut choisir que des valeurs connues du modèle.
# =============================================================================
def creer_champ(key, f, art):
    """Crée le bon champ du formulaire (liste déroulante ou champ numérique)."""
    wkey = f"{key}_{f['name']}"
 
    if f["kind"] == "cat":                                   # variable catégorielle
        options = art["encoders"][f["enc"]].classes_.tolist()
        idx = options.index(f["default"]) if f.get("default") in options else 0
        return st.selectbox(f["label"], options, index=idx, key=wkey)
 
    cast = int if f.get("int") else float                    # variable numérique
    kwargs = dict(min_value=cast(f["min"]), value=cast(f["default"]),
                  step=cast(f["step"]), key=wkey)
    if f.get("max") is not None:
        kwargs["max_value"] = cast(f["max"])
    return st.number_input(f["label"], **kwargs)
 
 
def onglet_simple(key, cfg, art):
    st.write(cfg["description"])
    feats = cfg["features"]
    moitie = (len(feats) + 1) // 2
 
    # Formulaire sur deux colonnes
    with st.form(f"{key}_form"):
        col1, col2 = st.columns(2)
        valeurs = {}
        for i, f in enumerate(feats):
            with (col1 if i < moitie else col2):
                valeurs[f["name"]] = creer_champ(key, f, art)
        envoyer = st.form_submit_button("🔮 Prédire", use_container_width=True)
 
    if not envoyer:
        return
 
    # Prédiction sur UNE ligne
    preds, proba = predire(cfg, art, pd.DataFrame([valeurs]))
 
    if cfg["task"] == "regression":
        unite = cfg.get("unit", "")
        carte_resultat(cfg["target_label"], f"{preds[0]:,.2f} {unite}".strip())
    else:
        carte_resultat(cfg["target_label"], preds[0])
        st.markdown("**Probabilité par classe**")
        for classe, p in proba.iloc[0].sort_values(ascending=False).items():
            st.caption(f"{classe} : {p:.1%}")
            st.progress(float(p))
 
 
# =============================================================================
# ÉTAPE 7 : ONGLET 2 : PRÉDICTION MULTIPLE (fichier CSV)
# Les lignes qui contiennent une valeur manquante (None / vide) sont
# simplement ignorées, sans message.
# =============================================================================
def prediction_multiple(cfg, art, df):
    """Prédit toutes les lignes complètes du CSV. Retourne (résultat, erreur)."""
    names = [f["name"] for f in cfg["features"]]
    data = cfg["prep"](df.copy()) if cfg.get("prep") else df.copy()
 
    # Vérifier que toutes les colonnes attendues sont présentes
    absentes = [n for n in names if n not in data.columns]
    if absentes:
        return None, "Colonnes manquantes dans le fichier : " + ", ".join(absentes)
 
    # Les valeurs non numériques dans une colonne numérique deviennent « vides »
    for f in cfg["features"]:
        if f["kind"] == "num":
            data[f["name"]] = pd.to_numeric(data[f["name"]], errors="coerce")
 
    # >>> On IGNORE les lignes avec des valeurs manquantes (sans message) <<<
    data = data.dropna(subset=names)
    if data.empty:
        return None, "Aucune ligne exploitable dans ce fichier."
 
    # Vérifier que les catégories sont connues du modèle
    for f in cfg["features"]:
        if f["kind"] == "cat":
            connues = set(art["encoders"][f["enc"]].classes_)
            inconnues = sorted(set(data[f["name"]].astype(str).str.strip()) - connues)
            if inconnues:
                return None, (f"Valeurs inconnues dans « {f['name']} » : "
                              + ", ".join(inconnues[:10]) + (" …" if len(inconnues) > 10 else ""))
 
    # Prédire, puis ajouter les colonnes de résultat aux lignes conservées
    preds, proba = predire(cfg, art, data)
    sortie = df.loc[data.index].copy()
    sortie[f"{cfg['target']}_predit"] = preds
    if proba is not None:
        sortie["Confiance"] = proba.max(axis=1).round(3).to_numpy()
    return sortie, None
 
 
def onglet_multiple(key, cfg, art):
    st.write(cfg["description"])
    names = [f["name"] for f in cfg["features"]]
    st.write("Colonnes attendues dans le CSV : " + ", ".join(f"`{n}`" for n in names))
    if cfg.get("csv_note"):
        st.caption(cfg["csv_note"])
 
    # Fichier modèle à télécharger (en-têtes seulement)
    st.download_button(
        "📄 Télécharger un modèle de CSV",
        data=pd.DataFrame(columns=names).to_csv(index=False).encode("utf-8"),
        file_name=f"modele_{key}.csv",
        mime="text/csv",
        key=f"{key}_template",
    )
 
    # Importer le fichier (sep=None : détecte « , » ou « ; » automatiquement)
    fichier = st.file_uploader("Importer un fichier CSV", type=["csv"], key=f"{key}_file")
    if fichier is None:
        return
 
    df = pd.read_csv(fichier, sep=None, engine="python")
    st.write("Aperçu des données chargées :")
    st.dataframe(df.head(), use_container_width=True)
 
    res_key = f"{key}_result"
    if st.button("🔮 Prédire", key=f"{key}_btn"):
        sortie, erreur = prediction_multiple(cfg, art, df)
        if erreur:
            st.session_state.pop(res_key, None)
            st.error(erreur)
        else:
            # On garde le résultat en mémoire pour qu'il reste affiché
            st.session_state[res_key] = (fichier.name, sortie)
 
    sauve = st.session_state.get(res_key)
    if sauve and sauve[0] == fichier.name:
        st.write("Résultats des prédictions :")
        st.dataframe(sauve[1], use_container_width=True)
        st.download_button(
            "⬇️ Télécharger les prédictions (CSV)",
            data=sauve[1].to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_{key}.csv",
            mime="text/csv",
            key=f"{key}_download",
        )
 
 
# =============================================================================
# ÉTAPE 8 : INTERFACE : TITRE, BARRE LATÉRALE ET ONGLETS
# =============================================================================
def afficher_page(key, cfg):
    """Affiche une application complète : chargement du modèle + 2 onglets."""
    st.subheader(f"{cfg['icon']} {cfg['title']}")
 
    # Charger le modèle (message clair si un fichier est introuvable)
    try:
        art = load_artifacts(cfg["folder"], tuple(cfg.get("scalers", [])))
    except FileNotFoundError as e:
        nom = Path(e.filename).name if e.filename else "rf_model.joblib"
        st.error(f"Fichier introuvable : `{nom}` dans `{find_model_dir(cfg['folder']).name}/`. "
                 "Exportez-le depuis le notebook puis ajoutez-le au dépôt.")
        return
 
    probleme = check_artifacts(cfg, art)
    if probleme:
        st.error(probleme)
        return
 
    # Les deux onglets
    tab1, tab2 = st.tabs(["Prédiction simple", " Prédiction multiple"])
    with tab1:
        onglet_simple(key, cfg, art)
    with tab2:
        onglet_multiple(key, cfg, art)
 
 
# --- Page active (mémorisée entre deux clics) ---
if "page" not in st.session_state:
    st.session_state.page = next(iter(PAGES))
 
# --- Barre latérale : un bouton par application ---
with st.sidebar:
    st.header("Applications")
    for nom, cfg in PAGES.items():
        active = st.session_state.page == nom
        if st.button(f"{cfg['icon']}  {nom}", key=f"nav_{nom}",
                     type="primary" if active else "secondary"):
            st.session_state.page = nom
            st.rerun()
    st.divider()
    st.caption("Modèles Random Forest entraînés sous Python (scikit-learn).")
 
# --- Bandeau d'en-tête ---
st.markdown(
    """
    <div class="hero">
        <h1>Welcome to my prediction app</h1>
        <p>Quatre modèles de Machine Learning : prédiction simple ou par fichier CSV.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
 
# --- Contenu : l'application choisie ---
page = st.session_state.page
afficher_page(page.lower().replace(" ", "_"), PAGES[page])
 
