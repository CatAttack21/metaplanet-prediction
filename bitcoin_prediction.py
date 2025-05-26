import pandas as pd
import numpy as np
from datetime import datetime
from scipy import optimize

def btc_power_law_formula(index):
    """Bitcoin Power Law model"""
    genesis = pd.Timestamp('2009-01-03')
    days_since_genesis = (index - genesis).days.values.astype(float)
    days_since_genesis[days_since_genesis < 1] = 1
    price = 10**-17 * (days_since_genesis ** 5.92)
    support = 0.5 * price
    resistance = 4.0 * price
    return support, price, resistance

def weierstrass_function(t, a=0.5, b=3, n_terms=10):
    """Generate Weierstrass-like function"""
    w = np.zeros_like(t, dtype=float)
    for n in range(n_terms):
        w += a**n * np.cos(np.pi * b**n * t)
    w = w / np.max(np.abs(w))
    return w

def multi_weierstrass(t, configs):
    """Overlay multiple Weierstrass functions with time-based amplitude"""
    wsum = np.zeros_like(t, dtype=float)
    for cfg in configs:
        w = weierstrass_function(
            t * cfg.get('scale', 1.0),
            a=cfg.get('a', 0.5),
            b=cfg.get('b', 3),
            n_terms=cfg.get('n_terms', 10)
        )
        wsum += cfg.get('weight', 1.0) * w
    
    wsum = wsum / np.max(np.abs(wsum))
    # Apply time-based amplitude scaling
    wsum *= get_time_based_amplitude(t)
    return wsum

def get_time_based_amplitude(t):
    """Calculate time-based amplitude scaling that transitions from 50% to 20%"""
    start_date = np.datetime64('2025-01-01')
    end_date = np.datetime64('2030-01-01')
    start_amp = 0.5  # 50% swing
    end_amp = 0.2    # 20% swing
    
    # Convert numeric days to datetime
    dates = pd.to_datetime('2009-01-03') + pd.to_timedelta(t, unit='D')
    scaling = np.ones_like(t, dtype=float) * end_amp
    
    mask = dates < end_date
    time_fraction = (dates[mask] - start_date) / (end_date - start_date)
    time_fraction = np.clip(time_fraction, 0, 1)
    scaling[mask] = start_amp - (start_amp - end_amp) * time_fraction
    scaling[dates < start_date] = start_amp
    
    return scaling

def predict_bitcoin_prices(start_date, end_date, last_price):
    """Predict Bitcoin prices using combined power law and Weierstrass models with smooth transition"""
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    future_df = pd.DataFrame(index=future_dates)
    
    _, future_center, _ = btc_power_law_formula(future_dates)
    
    t = np.arange(len(future_df))
    configs = [
        {'a': 0.5, 'b': 10, 'n_terms': 10, 'weight': 3.0, 'scale': 1/1460},
        {'a': 0.2, 'b': 10, 'n_terms': 10, 'weight': 1.0, 'scale': 1/365},
        {'a': 0.3, 'b': 5, 'n_terms': 20, 'weight': 2.0, 'scale': 1/1800},
        {'a': 0.3, 'b': 5, 'n_terms': 20, 'weight': 1.5, 'scale': 1/90},
        {'a': 0.5, 'b': 3, 'n_terms': 30, 'weight': 1.0, 'scale': 1/30}
    ]
    w = multi_weierstrass(t, configs)
    
    # Initialize price array
    prices = np.zeros(len(future_df))
    initial_price = float(last_price.iloc[0]) if isinstance(last_price, pd.Series) else float(last_price)
    prices[0] = initial_price

    # Calculate initial trend using linear regression on last 30 days
    transition_days = 60  # Extended transition period
    if isinstance(last_price, pd.Series) and len(last_price) >= 60:
        X = np.arange(60).reshape(-1, 1)
        y = last_price[-60:].values
        reg = optimize.minimize(
            lambda x: np.sum((y - (x[0] * X.flatten() + x[1]))**2),
            [0, initial_price],
            method='Nelder-Mead'
        ).x
        initial_trend = reg[0]  # Daily price change
    else:
        initial_trend = 0

    # Smooth transition period (270 days with exponential easing)
    for i in range(1, len(future_df)):
        if i < transition_days:
            # Calculate exponential transition
            power_law_price = future_center[i]
            price_diff = power_law_price - initial_price
            
            # Use exponential easing function
            progress = i / transition_days
            ease_factor = 1 - np.exp(-4 * progress)  # Exponential ease-in
            base_price = initial_price + price_diff * ease_factor
            
            # Add Weierstrass oscillation with reducing amplitude
            decay_factor = 1 - ease_factor
            weierstrass_component = w[i] * price_diff * decay_factor * 0.15
            prices[i] = base_price + weierstrass_component
        else:
            # More power law influence but maintain some volatility
            base_price = future_center[i]
            osc = w[i] * 0.5  # Reduced from 0.55
            amplitude = 1.0 * base_price  # Reduced from 0.44
            prices[i] = base_price + osc * amplitude

        # Ensure no negative prices and limit daily changes
        max_daily_change = 0.25  # Reduced from 0.22 for smoother transitions
        if i > 0:
            min_price = prices[i-1] * (1 - max_daily_change)
            max_price = prices[i-1] * (1 + max_daily_change)
            prices[i] = np.clip(prices[i], min_price, max_price)
    
    future_df['Price'] = prices
    future_df['CAGR'] = (future_df['Price'] / future_df['Price'].shift(365)) ** (1 / 1) - 1
    
    return future_df
