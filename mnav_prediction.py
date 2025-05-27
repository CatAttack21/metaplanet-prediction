import numpy as np
import pandas as pd
from datetime import datetime

def calculate_mnav_with_volatility(btc_value, days_from_start, base_volatility=0.12):
    """
    Calculates mNAV with enhanced volatility and mean reversion
    Returns: Float with calculated mNAV value including volatility
    """
    # Calculate power law baseline (theoretical fair value)
    theoretical_mcap = 35.1221 * (btc_value ** 0.89)
    power_law_mnav = theoretical_mcap / btc_value

    # Add random overshooting for mean reversion targets
    if np.random.random() < 0.15:  # 15% chance of setting new overshoot target
        # Generate asymmetric overshoots
        if np.random.random() < 0.5:  # Upside overshoot
            overshoot = np.random.uniform(1.33, 4.0)
        else:  # Downside overshoot
            overshoot = np.random.uniform(0.5, 0.8)
        target_mnav = power_law_mnav * overshoot
    else:
        target_mnav = power_law_mnav

    # Calculate volatility and noise
    volatility = base_volatility * (1 + 0.5 * np.sin(days_from_start / 30))
    noise = np.random.normal(0, volatility)
    
    # Get previous mNAV (or use power law if first calculation)
    current_mnav = getattr(calculate_mnav_with_volatility, 'last_mnav', power_law_mnav)
    
    # Mean reversion strength varies randomly
    reversion_speed = np.random.uniform(0.05, 0.15)
    
    # Calculate new mNAV with mean reversion and noise
    new_mnav = current_mnav + (target_mnav - current_mnav) * reversion_speed + noise
    
    # Apply minimum mNAV floor
    min_mnav = power_law_mnav * 0.4
    new_mnav = max(min_mnav, new_mnav)
    
    # Store for next calculation
    calculate_mnav_with_volatility.last_mnav = new_mnav
    
    return new_mnav
