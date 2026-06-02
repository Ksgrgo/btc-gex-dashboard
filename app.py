import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm
import matplotlib.pyplot as plt
import time

# Oldal beállítása
st.set_page_config(page_title="BTC Deribit GEX Option B View", layout="wide")

# Automatikus frissítés kapcsoló a bal oldalsávban
auto_refresh = st.sidebar.checkbox("Automatikus frissítés (60 másodpercenként)", value=True)

def load_and_calculate():
    try:
        # 1. Aktuális BTC Spot ár lekérése
        index_res = requests.get("https://deribit.com/api/v2/public/get_index_price?index_name=btc_usd").json()
        spot_price = index_res['result']['index_price']
        
        # 2. Nyers opciós lánc adatok lekérése
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
        opt_type = parts[3]
        
        try:
            expiry_date = datetime.strptime(expiry_str, '%d%b%y').replace(hour=8, minute=0, second=0)
        except Exception:
            continue
            
        time_to_expiry = expiry_date - now
        
        # Szűrés szigorúan maximum 365 napra (<= 365.0 DTE)
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
    # Gamma és GEX számítás
    df['gamma'] = [calc_gamma(spot_price, r['strike'], r['T'], r['iv']) for _, r in df.iterrows()]
    df['gex'] = df.apply(
        lambda r: r['oi'] * r['gamma'] * (spot_price ** 2) if r['type'] == 'C' else -r['oi'] * r['gamma'] * (spot_price ** 2),
        axis=1
    )
    
    # Strike árak szűrése a Spot körül az azonos nézetért (+-25%)
    min_strike = spot_price * 0.75
    max_strike = spot_price * 1.25
    df_filtered = df[(df['strike'] >= min_strike) & (df['strike'] <= max_strike)]
    
    strike_summary = df_filtered.groupby('strike')['gex'].sum().reset_index().sort_values('strike')
    strike_summary['cumulative_gex'] = strike_summary['gex'].cumsum()
    
    # Gamma Flip kiszámítása (ahol a kumulatív érték keresztezi a 0-át)
    flip_price = None
    cum_vals = strike_summary['cumulative_gex'].values
    strikes = strike_summary['strike'].values
    idx = np.where(np.diff(np.sign(cum_vals)))[0]
    if len(idx) > 0:
        i = idx[0]
        x1, x2 = strikes[i], strikes[i+1]
        y1, y2 = cum_vals[i], cum_vals[i+1]
        flip_price = x1 - y1 * (x2 - x1) / (y2 - y1)

    # --- MATPLOTLIB GRAFIKON ÉPÍTÉSE (Hajszálpontos másolat) ---
    plt.rcParams['figure.facecolor'] = 'white'
    fig, ax1 = plt.subplots(figsize=(12, 6.5))
    
    # Háttér és rács kikapcsolása / minimalizálása, mint a képen
    ax1.set_facecolor('white')
    
    # Dinamikus oszlopszélesség a szép megjelenésért
    bar_width = 400 if len(strike_summary) > 0 else 500
    
    # 1. Bal oldali Y tengely: Bars (Near-expiry GEX)
    # Szín: az a jellegzetes matt steel-blue / világoskék
    bar_color = '#79add2'
    bars = ax1.bar(strike_summary['strike'], strike_summary['gex'], width=bar_width, 
                  color=bar_color, alpha=0.75, edgecolor=bar_color, linewidth=0.5)
    
    ax1.set_xlabel('Strike', fontsize=10, labelpad=8)
    ax1.set_ylabel('Near-expiry GEX by strike', fontsize=10, labelpad=8)
    
    # 2. Jobb oldali Y tengely: Line (Cumulative near-expiry GEX)
    ax2 = ax1.twinx()
    line_color = '#2b7bba'
    line, = ax2.plot(strike_summary['strike'], strike_summary['cumulative_gex'], 
                     color=line_color, linewidth=1.8)
    ax2.set_ylabel('Cumulative GEX', fontsize=10, labelpad=8)
    
    # Vízszintes nullvonal
    ax1.axhline(0, color='#4682b4', linewidth=0.8, alpha=0.7)
    
    # Függőleges Spot vonal
    spot_line = ax1.axvline(spot_price, color='#2b7bba', linewidth=1.0, alpha=0.9)
    
    # Intraday Gamma Flip sötétkék pötty a nullvonalon
    flip_dot = None
    if flip_price:
        flip_dot, = ax2.plot(flip_price, 0, marker='o', color='#005a9c', 
                             markersize=9, linestyle='None')

    # Tengelyek formázása (Matplotlib automatikusan kirakja az 1e9 és 1e10 szorzókat felülre!)
    ax1.tick_params(axis='both', which='major', labelsize=9)
    ax2.tick_params(axis='y', which='major', labelsize=9)
    
    # Jelmagyarázat (Legend) manuális felépítése pontos sorrendben és feliratokkal
    handles = [spot_line, bars, line]
    labels = [
        f"Spot {spot_price:,.0f}".replace(",", ""),
        "Near-expiry GEX (<= 365.0 DTE)",
        "Cumulative near-expiry GEX"
    ]
    
    if flip_dot:
        handles.append(flip_dot)
        labels.append(f"Intraday gamma flip {flip_price:,.0f}".replace(",", ""))
        
    ax1.legend(handles, labels, loc='upper left', frameon=True, 
               facecolor='white', edgecolor='#d3d3d3', fontsize=9.5)
    
    # Cím és alcím beállítása
    plt.title("BTC Deribit GEX Option B View\nBars = near-expiry GEX (<= 365.0 DTE) | Line = cumulative near-expiry", 
              fontsize=11, pad=15, linespacing=1.3)
    
    # Szoros illeszkedés a szélekhez
    plt.tight_layout()
    
    # Megjelenítés Streamlitben static képként (így a design fix marad)
    st.pyplot(fig)

st.caption(f"Utolsó frissítés (UTC): {datetime.utcnow().strftime('%H:%M:%S')}")

if auto_refresh:
    time.sleep(60)
    st.rerun()
