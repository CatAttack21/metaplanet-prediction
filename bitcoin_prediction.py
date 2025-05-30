import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from volatility_analysis import (
    calculate_historical_volatility,
    fit_power_law_params,
    calculate_support_resistance,
    generate_realistic_noise
)

def btc_power_law_formula(index):
    """Bitcoin Power Law model"""
    genesis = pd.Timestamp('2009-01-03')
    days_since_genesis = (index - genesis).days.values.astype(float)
    days_since_genesis[days_since_genesis < 1] = 1
    price = 10**-17 * (days_since_genesis ** 5.95)
    support = 0.5 * price
    resistance = 4.0 * price
    return support, price, resistance

def predict_bitcoin_prices(start_date, end_date, last_price, historical_data=None):
    """
    Predict Bitcoin prices using power law as baseline with oscillating swings
    """
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Initialize predictions DataFrame
    predictions = pd.DataFrame(index=dates)
    predictions['Price'] = 0.0
    
    # Get power law model values
    support, baseline, resistance = btc_power_law_formula(dates)
    
    # Scale to match last known price
    initial_price = float(last_price.iloc[0] if isinstance(last_price, pd.Series) else last_price)
    scale_factor = initial_price / baseline[0]
    
    # Scale all curves
    baseline = baseline * scale_factor
    support = support * scale_factor
    resistance = resistance * scale_factor
    
    # Set initial price
    predictions.at[dates[0], 'Price'] = initial_price
    
    # Generate combined oscillation pattern
    time_index = np.arange(len(dates))
    combined_oscillation = np.zeros(len(dates))
    
    # Multiple frequency components
    periods = [1, 5, 10, 30, 90, 180, 360, 1460]  # Short, medium, and long cycles
    amplitudes = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09, 0.1]  # Increasing impact for longer cycles
    
    for period, amplitude in zip(periods, amplitudes):
        cycle = 2 * np.pi * time_index / period
        combined_oscillation += amplitude * np.sin(cycle)
    
    # Add noise component
    noise = generate_realistic_noise(len(dates), 0.1)  # Reduced volatility
    
    # Calculate prices respecting support/resistance
    for i in range(1, len(dates)):
        # Calculate oscillation factor
        oscillation = 1.0 + combined_oscillation[i] + noise[i]
        
        # Calculate price from baseline and oscillation
        new_price = baseline[i] * oscillation
        
        # Ensure price stays within support/resistance
        new_price = np.clip(new_price, support[i], resistance[i])
        
        # Limit daily moves to maintain realism
        prev_price = predictions.at[dates[i-1], 'Price']
        max_daily_change = 0.15  # 15% maximum daily change
        max_up = prev_price * (1 + max_daily_change)
        max_down = prev_price * (1 - max_daily_change)
        new_price = np.clip(new_price, max_down, max_up)
        
        predictions.at[dates[i], 'Price'] = new_price

    return predictions['Price']
