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
st.set_page_config(page_title="BTC Deribit GEX Option B View", layout="wide", page_icon="📈")

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
            expiry_date = datetime.strptime(expiry_str, '%d%b%y').replace(hour=8, minute=0, second=0)
        except Exception:
            continue
            
        time_to_expiry = expiry_date - now
        
        # SZŰRÉS: Csak a 0 és 365 nap közötti lejáratok (<= 365.0 DTE)
        if time_to_expiry.days < 0 or time_to_expiry.days > 365:
            continue
            
        T = time_to_expiry.total_seconds() / (365 * 24 * 3600)
        
        rows.append({
            'strike': strike,
            'type': opt_type,
            'oi': oi,
            'iv': iv / 100.0,
            'T': T
        })
        
    if not rows:
        return None, spot_price
        
    df = pd.DataFrame(rows)
    return df, spot_price

def calc_gamma(S, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

df, spot_price = load_and_calculate()

if df is not None:
    df['gamma'] = [calc_gamma(spot_price, r['strike'], r['T'], r['iv']) for _, r in df.iterrows()]
    df['gex'] = df.apply(
        lambda r: r['oi'] * r['gamma'] * (spot_price ** 2) if r['type'] == 'C' else -r['oi'] * r['gamma'] * (spot_price ** 2),
        axis=1
    )
    
    # Strike-ok szűrése a spot körül (+- 25% a szép láthatóságért, mint a képen)
    min_strike = spot_price * 0.75
    max_strike = spot_price * 1.25
    df_filtered = df[(df['strike'] >= min_strike) & (df['strike'] <= max_strike)]
    
    strike_summary = df_filtered.groupby('strike')['gex'].sum().reset_index().sort_values('strike')
    
    # Kumulált GEX kiszámítása a strike árak mentén haladva
    strike_summary['cumulative_gex'] = strike_summary['gex'].cumsum()
    
    # Skálázás pontosan a kép szerint: Bal oldal -> 1e9 (milliárd), Jobb oldal -> 1e10 (10 milliárd)
    strike_summary['gex_b'] = strike_summary['gex'] / 1e9
    strike_summary['cumulative_gex_b10'] = strike_summary['cumulative_gex'] / 1e10
    
    # Gamma Flip meghatározása ott, ahol a kumulált vonal keresztezi a 0-t
    flip_price = None
    cum_vals = strike_summary['cumulative_gex'].values
    strikes = strike_summary['strike'].values
    idx = np.where(np.diff(np.sign(cum_vals)))[0]
    if len(idx) > 0:
        i = idx[0]
        x1, x2 = strikes[i], strikes[i+1]
        y1, y2 = cum_vals[i], cum_vals[i+1]
        flip_price = x1 - y1 * (x2 - x1) / (y2 - y1) # Lineáris interpoláció a pontos ponthoz
    
    # Grafikon felépítése fehér háttérrel
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 1. Oszlopok: Egységes világoskék (Near-expiry GEX)
    fig.add_trace(
        go.Bar(
            x=strike_summary['strike'],
            y=strike_summary['gex_b'],
            name=f"Near-expiry GEX (<= 365.0 DTE)",
            marker_color='rgba(143, 186, 217, 0.75)', # Pontosan az a matt világoskék
            marker_line=dict(color='rgba(143, 186, 217, 1)', width=0.5),
            opacity=0.9
        ),
        secondary_y=False
    )
    
    # 2. Vonal: Kumulált sötétebb kék vonal (Cumulative near-expiry GEX)
    fig.add_trace(
        go.Scatter(
            x=strike_summary['strike'],
            y=strike_summary['cumulative_gex_b10'],
            name="Cumulative near-expiry GEX",
            line=dict(color='#2B7BBA', width=2.5),
            mode='lines'
        ),
        secondary_y=True
    )
    
    # 3. Függőleges vonal a Spot árnak
    fig.add_vline(
        x=spot_price, 
        line_width=1.5, 
        line_color="#1F4E79", 
        name=f"Spot {spot_price:,.0f}"
    )
    
    # Dummy trace-ek csak azért, hogy a jelmagyarázat (Legend) pontosan úgy nézzen ki, mint a képen
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#1F4E79', width=1.5), name=f"Spot {spot_price:,.0f}"), secondary_y=False)
    
    # 4. A Gamma Flip PÖTTY a kumulált vonalon (ha létezik a tartományban)
    if flip_price:
        fig.add_trace(
            go.Scatter(
                x=[flip_price],
                y=[0], # A nullvonalon metszi egymást
                mode='markers',
                marker=dict(color='#005A9C', size=11, symbol='circle'),
                name=f"Intraday gamma flip {flip_price:,.0f}"
            ),
            secondary_y=True
        )
        
    # Vízszintes nullvonal (X tengely kiemelése)
    fig.add_hline(y=0, line_width=1, line_color="#7F7F7F")
    
    # Dizájn beállítások: Fehér háttér, pontos feliratok
    fig.update_layout(
        title={
            'text': "<b>BTC Deribit GEX Option B View</b><br><span style='font-size:13px;color:gray;'>Bars = near-expiry GEX (<= 365.0 DTE) | Line = cumulative near-expiry</span>",
            'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
        },
        xaxis_title="Strike",
        template="plotly_white",
        height=700,
        grid=dict(rows=1, columns=1),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        ),
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    # Tengelyek feliratai és formázása (1e9 és 1e10 jelölések imitálása)
    fig.update_yaxes(title_text="Near-expiry GEX by strike (x10⁹)", secondary_y=False, showgrid=True, gridcolor='#E5E5E5')
    fig.update_yaxes(title_text="Cumulative GEX (x10¹⁰)", secondary_y=True, showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor='#E5E5E5', tickformat=",.0f")
    
    # Megjelenítés a böngészőben
    st.plotly_chart(fig, use_container_width=True)

st.caption(f"Utolsó frissítés (UTC): {datetime.utcnow().strftime('%H:%M:%S')}")

if auto_refresh:
    time.sleep(60)
    st.rerun()
