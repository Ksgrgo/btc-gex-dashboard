import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Oldal beállítása
st.set_page_config(page_title="BTC Deribit GEX Live Terminal", layout="wide", page_icon="📈")

st.title("📈 BTC Deribit GEX Opció-Profil (Élő Adatok)")
st.write("Ez az alkalmazás a nyers Open Interest (OI) adatokból becsli a GEX-et, súlyozás nélkül.")

# Automatikus frissítés kapcsoló a bal oldalsávban
auto_refresh = st.sidebar.checkbox("Automatikus frissítés (60 másodpercenként)", value=True)

def load_and_calculate():
    try:
        # 1. Aktuális BTC Spot ár lekérése
        index_res = requests.get("https://deribit.com/api/v2/public/get_index_price?index_name=btc_usd").json()
        spot_price = index_res['result']['index_price']
        
        # 2. Nyers opciós lánc adatok lekérése (nyitott szerződésállomány)
        options_res = requests.get("https://deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option").json()
        data = options_res['result']
    except Exception as e:
        st.error(f"Hiba az adatok letöltése közben: {e}")
        return None, None
        
    rows = []
    now = datetime.utcnow()
    
    for item in data:
        name = item['instrument_name']
        oi = item.get('open_interest', 0)
        iv = item.get('mark_iv', 0)
        
        if oi == 0 or iv == 0:
            continue
            
        parts = name.split('-')
        if len(parts) < 4:
            continue
            
        expiry_str = parts[1]
        strike = float(parts[2])
        opt_type = parts[3] # 'C' = Call, 'P' = Put
        
        try:
            # Deribit formátum feldolgozása (pl. 26JUN26)
            expiry_date = datetime.strptime(expiry_str, '%d%b%y').replace(hour=8, minute=0, second=0)
        except Exception:
            continue
            
        time_to_expiry = expiry_date - now
        T = time_to_expiry.total_seconds() / (365 * 24 * 3600) # Évben kifejezve
        
        if T <= 0:
            continue
            
        rows.append({
            'strike': strike,
            'type': opt_type,
            'oi': oi,
            'iv': iv / 100.0, # Százalékból tizedestört
            'T': T
        })
        
    if not rows:
        st.warning("Nem sikerült feldolgozni az opciós adatokat.")
        return None, spot_price
        
    df = pd.DataFrame(rows)
    return df, spot_price

def calc_gamma(S, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

def find_gamma_flip(df_input, current_spot):
    # Leteszteljük a szinteket a spot ár környezetében (-20% és +20% között)
    prices = np.linspace(current_spot * 0.8, current_spot * 1.2, 200)
    net_gex_values = []
    
    T = df_input['T'].values
    K = df_input['strike'].values
    sigma = df_input['iv'].values
    oi = df_input['oi'].values
    is_call = (df_input['type'] == 'C').values
    
    for p in prices:
        d1 = (np.log(p / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (p * sigma * np.sqrt(T))
        gex = oi * gamma * (p ** 2)
        # Call opció pozitív, Put opció negatív előjelű a standard modellben
        gex = np.where(is_call, gex, -gex)
        net_gex_values.append(np.sum(gex))
        
    net_gex_values = np.array(net_gex_values)
    idx = np.where(np.diff(np.sign(net_gex_values)))[0]
    if len(idx) > 0:
        i = idx[0]
        p1, p2 = prices[i], prices[i+1]
        g1, g2 = net_gex_values[i], net_gex_values[i+1]
        return p1 - g1 * (p2 - p1) / (g2 - g1)
    return None

df, spot_price = load_and_calculate()

if df is not None:
    # Gamma és GEX számítás az aktuális spot ár mellett
    df['gamma'] = [calc_gamma(spot_price, r['strike'], r['T'], r['iv']) for _, r in df.iterrows()]
    df['gex'] = df.apply(
        lambda r: r['oi'] * r['gamma'] * (spot_price ** 2) if r['type'] == 'C' else -r['oi'] * r['gamma'] * (spot_price ** 2),
        axis=1
    )
    
    # Szűrés a spot ár körüli releváns tartományra (+- 25%), hogy jól látható legyen a grafikon
    min_strike = spot_price * 0.75
    max_strike = spot_price * 1.25
    df_filtered = df[(df['strike'] >= min_strike) & (df['strike'] <= max_strike)]
    
    # Aggregáció strike árak szerint
    strike_summary = df_filtered.groupby('strike')['gex'].sum().reset_index().sort_values('strike')
    strike_summary['cumulative_gex'] = strike_summary['gex'].cumsum()
    
    # Átváltás milliárd USD egységre (mint a képeden: 1e9)
    strike_summary['gex_b'] = strike_summary['gex'] / 1e9
    strike_summary['cumulative_gex_b'] = strike_summary['cumulative_gex'] / 1e9
    
    flip_price = find_gamma_flip(df, spot_price)
    
    # Metrikák megjelenítése a lap tetején
    col1, col2 = st.columns(2)
    col1.metric("Aktuális BTC Spot Ár", f"${spot_price:,.2f}")
    if flip_price:
        col2.metric("Becsült Gamma Flip Ár", f"${flip_price:,.2f}")
    else:
        col2.metric("Gamma Flip Ár", "Nem található a tartományban")
        
    # Grafikon felépítése (Két Y tengellyel)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Oszlopok (GEX per strike) - Pozitív zöld, negatív piros
    colors = ['#00CC96' if val >= 0 else '#EF553B' for val in strike_summary['gex_b']]
    
    fig.add_trace(
        go.Bar(
            x=strike_summary['strike'],
            y=strike_summary['gex_b'],
            name="GEX Strike-onként",
            marker_color=colors,
            opacity=0.8
        ),
        secondary_y=False
    )
    
    # Kumulált vonal (Cumulative GEX)
    fig.add_trace(
        go.Scatter(
            x=strike_summary['strike'],
            y=strike_summary['cumulative_gex_b'],
            name="Kumulált GEX",
            line=dict(color='#AB63FA', width=3),
            mode='lines'
        ),
        secondary_y=True
    )
    
    # Függőleges vonalak az áraknak
    fig.add_vline(x=spot_price, line_width=2, line_dash="dash", line_color="#636EFA", annotation_text="Spot Ár")
    if flip_price:
        fig.add_vline(x=flip_price, line_width=2, line_dash="dot", line_color="#FFA15A", annotation_text="Gamma Flip")
        
    fig.update_layout(
        title="BTC Deribit GEX Opciós Profil (Minden Lejárat Összesítve)",
        xaxis_title="Strike Ár ($)",
        template="plotly_dark",
        height=650,
        legend=dict(x=0.01, y=0.99)
    )
    
    fig.update_yaxes(title_text="GEX Strike-onként (Milliárd $)", secondary_y=False)
    fig.update_yaxes(title_text="Kumulált GEX (Milliárd $)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

st.write(f"Utolsó frissítés (UTC): {datetime.utcnow().strftime('%H:%M:%S')}")

# Alvás és automatikus újrafuttatás
if auto_refresh:
    time.sleep(60)
    st.rerun()
