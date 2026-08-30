import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuration de la page
st.set_page_config(
    page_title="Canaux de Régression Boursière",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Canaux de Régression Linéaire & Écart-Type")
st.markdown("""
Analysez la tendance de long terme de **toutes les actions** (Marchés français `.PA`, US, ETF, etc.) avec les bandes de dispersion à **$\pm 1\sigma$** et **$\pm 2\sigma$**.
""")

# --- Barre latérale de configuration ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    
    ticker_input = st.text_input(
        "Ticker boursier :",
        value="OR.PA",
        help="Exemples : OR.PA (L'Oréal), MC.PA (LVMH), AI.PA (Air Liquide), TTE.PA (Total), AAPL (Apple), NVDA (Nvidia), SPY (S&P 500)"
    ).strip().upper()

    years_back = st.slider("Historique (années) :", min_value=3, max_value=40, value=15, step=1)
    
    model_type = st.radio(
        "Modèle de tendance :",
        ["Linéaire classique (y = a·x + b)", "Logarithmique / Exponentiel (rendement composé)"],
        index=0
    )
    
    use_adj_close = st.checkbox("Ajuster des dividendes & splits (Recommandé)", value=True)

# --- Fonction de récupération des données sécurisée ---
def get_clean_stock_data(symbol, years, use_adjusted=True):
    start_date = datetime.now() - timedelta(days=int(years * 365.25))
    
    # Téléchargement robuste
    df = yf.download(
        symbol, 
        start=start_date.strftime('%Y-%m-%d'),
        progress=False,
        auto_adjust=False
    )
    
    if df is None or df.empty:
        # Tentative avec l'objet Ticker en secours
        t = yf.Ticker(symbol)
        df = t.history(period=f"{years}y")
        if df.empty:
            return None

    # Nettoyage des colonnes (gère les MultiIndex de Yahoo Finance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Choix de la colonne de prix
    target_col = 'Adj Close' if (use_adjusted and 'Adj Close' in df.columns) else 'Close'
    if target_col not in df.columns:
        if 'Close' in df.columns:
            target_col = 'Close'
        else:
            return None

    # Extraction et nettoyage strict
    clean_series = df[target_col].dropna().astype(float)
    clean_series = clean_series[clean_series > 0] # supprime les valeurs aberrantes <= 0
    
    # Nettoyage des dates (suppression des fuseaux horaires pour alignement parfait)
    clean_series.index = pd.to_datetime(clean_series.index).tz_localize(None)
    
    return clean_series

# --- Traitement et affichage ---
if ticker_input:
    with st.spinner(f"Récupération des données pour {ticker_input}..."):
        prices = get_clean_stock_data(ticker_input, years_back, use_adj_close)

    if prices is None or len(prices) < 30:
        st.error(f"❌ Données insuffisantes pour le ticker **'{ticker_input}'**. Vérifiez le symbole (ex: ajoutez `.PA` pour Euronext Paris comme `MC.PA`, `OR.PA`).")
    else:
        # Dates et abscisses temporelles numériques (en jours depuis le début)
        dates = prices.index
        t0 = dates[0]
        x = np.array([(d - t0).days for d in dates], dtype=float)
        y = prices.values

        # --- Calcul mathématique sans faille (NumPy pur) ---
        if "Logarithmique" in model_type:
            log_y = np.log(y)
            slope, intercept = np.polyfit(x, log_y, 1)
            reg_log = slope * x + intercept
            residuals = log_y - reg_log
            std_dev = np.std(residuals)
            
            # Reconversion exponentielle
            reg_line = np.exp(reg_log)
            band_p1 = np.exp(reg_log + 1 * std_dev)
            band_m1 = np.exp(reg_log - 1 * std_dev)
            band_p2 = np.exp(reg_log + 2 * std_dev)
            band_m2 = np.exp(reg_log - 2 * std_dev)
            
            # Calcul R²
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            cagr = (np.exp(slope * 365.25) - 1) * 100
        else:
            slope, intercept = np.polyfit(x, y, 1)
            reg_line = slope * x + intercept
            residuals = y - reg_line
            std_dev = np.std(residuals)
            
            band_p1 = reg_line + 1 * std_dev
            band_m1 = reg_line - 1 * std_dev
            band_p2 = reg_line + 2 * std_dev
            band_m2 = reg_line - 2 * std_dev
            
            # Calcul R²
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            cagr = ((reg_line[-1] / reg_line[0]) ** (365.25 / (x[-1] - x[0])) - 1) * 100

        # Position actuelle du cours
        current_price = y[-1]
        current_reg = reg_line[-1]
        distance_to_reg = ((current_price - current_reg) / current_reg) * 100
        z_score = (residuals[-1]) / std_dev

        # --- Métriques en colonnes ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Dernier Cours", f"{current_price:.2f} €/$")
        col2.metric("Valeur Tendance (Régression)", f"{current_reg:.2f} €/$")
        col3.metric("Écart à la droite", f"{distance_to_reg:+.2f} %", delta=f"{distance_to_reg:+.2f}%")
        col4.metric("Qualité du modèle (R²)", f"{r_squared:.3f}", help="Plus proche de 1, plus la tendance est propre")

        # --- Graphique interactif Plotly ---
        fig = go.Figure()

        # Bandes 2 Sigmas (Extrêmes)
        fig.add_trace(go.Scatter(x=dates, y=band_p2, name="+2σ (Surévaluation extrême)", line=dict(color='rgba(255, 75, 75, 0.4)', dash='dot', width=1)))
        fig.add_trace(go.Scatter(x=dates, y=band_m2, name="-2σ (Sous-évaluation extrême)", line=dict(color='rgba(0, 204, 150, 0.4)', dash='dot', width=1), fill='tonexty', fillcolor='rgba(200, 200, 200, 0.05)'))

        # Bandes 1 Sigma (Modérées)
        fig.add_trace(go.Scatter(x=dates, y=band_p1, name="+1σ (Surévaluation)", line=dict(color='rgba(255, 165, 0, 0.7)', dash='dash', width=1.2)))
        fig.add_trace(go.Scatter(x=dates, y=band_m1, name="-1σ (Sous-évaluation)", line=dict(color='rgba(0, 150, 255, 0.7)', dash='dash', width=1.2)))

        # Droite centrale de régression
        fig.add_trace(go.Scatter(x=dates, y=reg_line, name="Tendance Moyenne (Régression)", line=dict(color='white', width=2.5)))

        # Cours de clôture réel
        fig.add_trace(go.Scatter(x=dates, y=y, name=f"Cours {ticker_input}", line=dict(color='#00F0FF', width=1.8)))

        # Mise en page du graphique
        fig.update_layout(
            template="plotly_dark",
            title=f"Canal de Régression sur {len(prices)} séances ({years_back} ans) - {ticker_input}",
            xaxis_title="Date",
            yaxis_title="Prix (€ / $)",
            height=650,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Interprétation financière instantanée ---
        st.subheader("💡 Diagnostic de valorisation :")
        if z_score > 2:
            st.error(f"⚠️ **Forte surévaluation (+{z_score:.2f}σ) :** Le cours est au-dessus de sa bande +2σ. Historiquement rare, risque élevé de correction.")
        elif z_score > 1:
            st.warning(f"🟡 **Légère surévaluation (+{z_score:.2f}σ) :** Le cours est au-dessus de sa droite moyenne.")
        elif z_score < -2:
            st.success(f"💎 **Forte sous-évaluation ({z_score:.2f}σ) :** Le cours est sous sa bande -2σ. Zone d'achat historique favorable.")
        elif z_score < -1:
            st.info(f"🔵 **Légère sous-évaluation ({z_score:.2f}σ) :** Le cours est sous sa moyenne de long terme.")
        else:
            st.write(f"⚪ **Neutre ({z_score:+.2f}σ) :** Le cours est proche de sa moyenne historique.")
