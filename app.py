"""Applications ML (Car data, Car dataset, Telephone, Taxi trips) — un seul fichier.
 
Lancer : streamlit run app.py
Modèles attendus dans models/<dossier>/ (voir PAGES plus bas).
"""
import re
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
import streamlit as st
 
# =============================================================================
# 1) CONFIGURATION DES 4 APPLICATIONS
# L'ordre de `features` = ordre des colonnes de x dans le notebook.
# `enc` = position de la colonne dans `encoders` / `uniques` (ordre des
# colonnes catégorielles du notebook).
# =============================================================================
def _split_adresse(df):
    """Car Data : le notebook découpe « Quartier,Ville » depuis la colonne Adresse."""
    if "Adresse" in df.columns:
        parts = df["Adresse"].astype(str).str.split(",", n=1, expand=True)
        df["Quartier"] = parts[0].str.strip()
        df["Ville"] = parts[1].str.strip() if parts.shape[1] > 1 else ""
        df = df.drop(columns="Adresse")
    return df
 
 
PAGES = {
    # ------------------------------------------------------------------ #
    "Car data": {
        "icon": "🚗",
        "title": "Car data — Voitures neuves / d'occasion (Dakar)",
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
        "icon": "🚙",
        "title": "Car dataset — Prix de vente d'une voiture",
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
        "icon": "📱",
        "title": "Telephone — État d'un téléphone",
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
        "icon": "🚕",
        "title": "Taxi trips — Prix d'une course",
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
# 2) MOTEUR : chargement des modèles, prédiction simple et multiple
# =============================================================================
# Chemins RELATIFS au projet (et non /content/... comme sur Colab).
# Les dossiers sont cherchés dans models/ puis à la racine du projet, sans tenir
# compte des majuscules, espaces, « _ » ou « - » (« Taxi trip » = « taxi_trip »).
APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"
 
 
def _norm(name: str) -> str:
    return re.sub(r"[\s_\-]", "", name.lower())
 
 
def find_model_dir(folder: str) -> Path:
    key = _norm(folder)
    for base in (MODELS_DIR, APP_DIR):
        if base.is_dir():
            for p in base.iterdir():
                if p.is_dir() and _norm(p.name) == key:
                    return p
    return MODELS_DIR / folder
 
 
# ----------------------------------------------------------------------------
# Chargement des objets sauvegardés (une seule fois grâce au cache)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts(folder: str, scaler_files: tuple):
    d = find_model_dir(folder)
    return {
        "model": joblib.load(d / "rf_model.joblib"),
        "encoders": joblib.load(d / "encoders.joblib"),
        "uniques": joblib.load(d / "uniques.joblib"),
        # les scalers sont appliqués dans l'ordre (ex. Robust puis MinMax)
        "scalers": [joblib.load(d / f) for f in scaler_files],
    }
 
 
# ----------------------------------------------------------------------------
# Validation + prédiction (vectorisées : pas de boucle ligne par ligne)
# ----------------------------------------------------------------------------
def validate_df(cfg, art, df):
    """Retourne un message d'erreur (str) ou None si tout est correct."""
    names = [f["name"] for f in cfg["features"]]
    missing = [n for n in names if n not in df.columns]
    if missing:
        return "Colonnes manquantes dans le fichier : " + ", ".join(missing)
    if df[names].isna().any().any():
        return "Le fichier contient des valeurs manquantes (ou une colonne entièrement vide) dans les colonnes utilisées. Cochez « Remplir automatiquement » ou complétez le fichier."
    for f in cfg["features"]:
        if f["kind"] == "cat":
            known = set(art["encoders"][f["enc"]].classes_)
            bad = sorted(set(df[f["name"]].astype(str).str.strip()) - known)
            if bad:
                return (f"Valeurs inconnues dans « {f['name']} » : "
                        + ", ".join(bad[:10]) + (" …" if len(bad) > 10 else ""))
    return None
 
 
def fill_missing(cfg, df):
    """Remplit les cases vides : médiane (nombres) / valeur la plus fréquente (texte)."""
    df = df.copy()
    n_filled = 0
    for f in cfg["features"]:
        col = f["name"]
        if col not in df.columns or not df[col].isna().any():
            continue
        if df[col].isna().all():
            continue  # colonne entièrement vide : impossible à remplir
        n_filled += int(df[col].isna().sum())
        if f["kind"] == "cat":
            df[col] = df[col].fillna(df[col].mode().iloc[0])
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
    return df, n_filled
 
 
def predict_df(cfg, art, df):
    """Retourne (prédictions, probabilités ou None)."""
    names = [f["name"] for f in cfg["features"]]
    X = df[names].copy()
    for f in cfg["features"]:
        if f["kind"] == "cat":
            enc = art["encoders"][f["enc"]]
            X[f["name"]] = enc.transform(X[f["name"]].astype(str).str.strip())
    X = X.apply(pd.to_numeric).to_numpy(dtype=float)
 
    # même normalisation que dans le notebook (sinon les prédictions sont fausses)
    for scaler in art["scalers"]:
        X = scaler.transform(X)
 
    model = art["model"]
    y = model.predict(X)
 
    if cfg["task"] == "regression":
        return np.round(y.astype(float), 2), None
 
    target_enc = art["encoders"][cfg["target_enc"]]
    labels = target_enc.inverse_transform(y.astype(int))
    proba = pd.DataFrame(
        model.predict_proba(X),
        columns=target_enc.inverse_transform(model.classes_.astype(int)),
    )
    return labels, proba
 
 
# ----------------------------------------------------------------------------
# Widgets
# ----------------------------------------------------------------------------
def check_artifacts(cfg, art):
    """Vérifie que le modèle du dossier correspond bien à cette page."""
    cats = [f["enc"] for f in cfg["features"] if f["kind"] == "cat"]
    if cfg["task"] == "classification":
        cats.append(cfg["target_enc"])
    need = max(cats) + 1 if cats else 0
    have = len(art["uniques"])
    if have < need or len(art["encoders"]) < need:
        return (f"Ce dossier contient {have} variable(s) catégorielle(s) mais cette page "
                f"en attend {need}. Ce n'est probablement pas le bon modèle : "
                "relancez le bon notebook avec le bon fichier CSV, puis réexportez.")
    for i in set(cats):
        if art["encoders"][i].classes_.dtype.kind in "iufb":
            return ("Les encodeurs contiennent des nombres au lieu des noms (marques, quartiers…). "
                    "Dans le notebook, relancez TOUT dans l'ordre (Exécution → Tout exécuter) : "
                    "les cellules uniques/encoders ne doivent pas être relancées après l'encodage. "
                    "Puis réexportez les .joblib.")
    n = getattr(art["model"], "n_features_in_", None)
    if n is not None and n != len(cfg["features"]):
        return (f"Le modèle attend {n} variables mais cette page en utilise "
                f"{len(cfg['features'])}. Ce n'est pas le bon modèle pour cette page.")
    return None
 
 
def _widget(key, f, art):
    wkey = f"{key}_{f['name']}"
    if f["kind"] == "cat":
        options = art["encoders"][f["enc"]].classes_.tolist()  # vrais noms (pas des nombres)
        idx = options.index(f["default"]) if f.get("default") in options else 0
        return st.selectbox(f["label"], options, index=idx, key=wkey)
 
    cast = int if f.get("int") else float
    kwargs = dict(
        min_value=cast(f["min"]),
        value=cast(f["default"]),
        step=cast(f["step"]),
        key=wkey,
    )
    if f.get("max") is not None:
        kwargs["max_value"] = cast(f["max"])
    return st.number_input(f["label"], **kwargs)
 
 
# ----------------------------------------------------------------------------
# Onglet 1 : prédiction simple
# ----------------------------------------------------------------------------
def simple_tab(key, cfg, art):
    st.write(cfg["description"])
    feats = cfg["features"]
    half = (len(feats) + 1) // 2
 
    with st.form(f"{key}_form"):
        col1, col2 = st.columns(2)
        values = {}
        for i, f in enumerate(feats):
            with (col1 if i < half else col2):
                values[f["name"]] = _widget(key, f, art)
        submit = st.form_submit_button("Prédire")
 
    if not submit:
        return
 
    row = pd.DataFrame([values])
    preds, proba = predict_df(cfg, art, row)
 
    if cfg["task"] == "regression":
        st.success(f"{cfg['target_label']} : {preds[0]:,.2f} {cfg.get('unit', '')}".strip())
    else:
        st.success(f"{cfg['target_label']} : **{preds[0]}**")
        st.write("Probabilité par classe :")
        st.dataframe((proba.T.rename(columns={0: "Probabilité"}) * 100).round(1))
 
 
# ----------------------------------------------------------------------------
# Onglet 2 : prédiction multiple (CSV)
# ----------------------------------------------------------------------------
def multiple_tab(key, cfg, art):
    st.write(cfg["description"])
    names = [f["name"] for f in cfg["features"]]
    st.write("Colonnes attendues dans le CSV : " + ", ".join(f"`{n}`" for n in names))
    if cfg.get("csv_note"):
        st.caption(cfg["csv_note"])
 
    st.download_button(
        "Télécharger un modèle de CSV",
        data=pd.DataFrame(columns=names).to_csv(index=False).encode("utf-8"),
        file_name=f"modele_{key}.csv",
        mime="text/csv",
        key=f"{key}_template",
    )
 
    file = st.file_uploader("Importer un fichier CSV", type=["csv"], key=f"{key}_file")
    if file is None:
        return
 
    # sep=None : détecte automatiquement « , » ou « ; » (CSV exportés d'Excel FR)
    df = pd.read_csv(file, sep=None, engine="python")
    st.write("Aperçu des données chargées :")
    st.dataframe(df.head())
 
    res_key = f"{key}_result"
    fill = st.checkbox(
        "Remplir automatiquement les valeurs manquantes "
        "(médiane pour les nombres, valeur la plus fréquente pour le texte)",
        value=True, key=f"{key}_fill",
    )
    if st.button("Prédire", key=f"{key}_btn"):
        data = cfg["prep"](df.copy()) if cfg.get("prep") else df.copy()
        n_filled = 0
        if fill:
            data, n_filled = fill_missing(cfg, data)
        error = validate_df(cfg, art, data)
        if error:
            st.session_state.pop(res_key, None)
            st.error(error)
        else:
            preds, proba = predict_df(cfg, art, data)
            out = df.copy()
            out[f"{cfg['target']}_predit"] = preds
            if proba is not None:
                out["Confiance"] = proba.max(axis=1).round(3).to_numpy()
            note = (f"{n_filled} valeur(s) manquante(s) remplie(s) automatiquement : "
                    "les prédictions de ces lignes sont approximatives."
                    if n_filled else "")
            st.session_state[res_key] = (file.name, out, note)
 
    saved = st.session_state.get(res_key)
    if saved and saved[0] == file.name:
        if saved[2]:
            st.warning(saved[2])
        st.write("Résultats des prédictions :")
        st.dataframe(saved[1])
        st.download_button(
            "Télécharger les prédictions (CSV)",
            data=saved[1].to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_{key}.csv",
            mime="text/csv",
            key=f"{key}_download",
        )
 
 
# ----------------------------------------------------------------------------
# Page complète
# ----------------------------------------------------------------------------
def render_page(key, cfg):
    st.subheader(f"{cfg['icon']} {cfg['title']}")
    try:
        art = load_artifacts(cfg["folder"], tuple(cfg.get("scalers", [])))
    except FileNotFoundError as e:
        st.error(
            f"Fichier de modèle introuvable : `{Path(e.filename).name if e.filename else 'rf_model.joblib'}` dans "
            f"`{find_model_dir(cfg['folder']).name}/`. Exportez-le depuis le notebook puis "
            "ajoutez-le au dépôt."
        )
        return
 
    problem = check_artifacts(cfg, art)
    if problem:
        st.error(problem)
        return
 
    tab1, tab2 = st.tabs(["Prédiction simple", "Prédiction multiple"])
    with tab1:
        simple_tab(key, cfg, art)
    with tab2:
        multiple_tab(key, cfg, art)
 
 
# =============================================================================
# 3) INTERFACE : titre + barre latérale + onglets
# =============================================================================
 
# ✏️ Mettez ici votre nom / le nom de votre application
APP_NAME = "Erica's ML App"
 
st.set_page_config(page_title=f"{APP_NAME} — Applications ML", page_icon="🤖", layout="wide")
 
# Boutons de la barre latérale sur toute la largeur (compatible toutes versions de Streamlit)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] .stButton > button { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)
 
# ------------------------- Page active -------------------------
if "page" not in st.session_state:
    st.session_state.page = next(iter(PAGES))
 
# ------------------------- Barre latérale (4 boutons) -------------------------
with st.sidebar:
    st.header("Applications")
    for name, cfg in PAGES.items():
        is_active = st.session_state.page == name
        if st.button(f"{cfg['icon']}  {name}", key=f"nav_{name}",
                     type="primary" if is_active else "secondary"):
            st.session_state.page = name
            st.rerun()
 
# ------------------------- En-tête : « Welcome to ... » -------------------------
st.title(f"Welcome to my prediction app")
st.divider()
 
# ------------------------- Contenu : onglets simple / multiple -------------------------
active = st.session_state.page
render_page(active.lower().replace(" ", "_"), PAGES[active])