"""
FRAM Assignment: Option Portfolio Management for Maruti Suzuki
Birla Institute of Technology and Science, Pilani
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew, kurtosis
from scipy.optimize import minimize_scalar
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)

print("="*80)
print("FRAM ASSIGNMENT: MARUTI SUZUKI OPTION PORTFOLIO ANALYSIS")
print("="*80)

# ============================================================================
# PART A: DATA COLLECTION & STATISTICS
# ============================================================================
print("\n" + "="*80)
print("PART A: DATA COLLECTION & STATISTICS")
print("="*80)

# Download Maruti stock data
ticker = "MARUTI.NS"
stock = yf.Ticker(ticker)

# Get 3 months of data
end_date = datetime.now()
start_date = end_date - timedelta(days=90)

print(f"\nDownloading data for {ticker}...")
df = yf.download(ticker, start=start_date, end=end_date, progress=False)

# Calculate returns and statistics
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
df = df.dropna()

# Summary statistics
daily_returns = df['Log_Return']
daily_std = daily_returns.std()
annualized_vol = daily_std * np.sqrt(252)
returns_skewness = skew(daily_returns)
returns_kurtosis = kurtosis(daily_returns)

# Current stock price
current_price = df['Close'].iloc[-1]

print("\n" + "-"*80)
print("SUMMARY STATISTICS")
print("-"*80)

summary_stats = pd.DataFrame({
    'Metric': ['Current Stock Price', 'Mean Daily Return', 'Daily Std Dev', 
               'Annualized Volatility', 'Skewness', 'Kurtosis'],
    'Value': [current_price, daily_returns.mean(), daily_std, 
              annualized_vol, returns_skewness, returns_kurtosis]
})
print(summary_stats.to_string(index=False))

# Plot stock price and returns
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(df.index, df['Close'], linewidth=2, color='blue')
axes[0].set_title('Maruti Stock Price (Last 3 Months)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Price (INR)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(df.index, daily_returns, linewidth=1, color='green')
axes[1].set_title('Daily Log Returns', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Return')
axes[1].set_xlabel('Date')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('part_a_stock_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: part_a_stock_analysis.png")

# ============================================================================
# PART B: OPTION PRICING (BLACK-SCHOLES-MERTON)
# ============================================================================
print("\n" + "="*80)
print("PART B: OPTION PRICING (BLACK-SCHOLES-MERTON)")
print("="*80)

# BSM Model Functions
def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    put = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return put

# Parameters
S0 = float(current_price)
r = 0.065  # Risk-free rate (approximate RBI repo rate)
sigma = annualized_vol

# Strike prices
ATM = round(S0 / 100) * 100  # Round to nearest 100
strikes = [
    ATM * 0.95,  # ATM - 5%
    ATM * 0.98,  # ATM - 2%
    ATM,         # ATM
    ATM * 1.02,  # ATM + 2%
    ATM * 1.05   # ATM + 5%
]

# Maturities (in years)
maturities_days = [30, 60, 90]
maturities = [days/365 for days in maturities_days]

print(f"\nUnderlying Price: ₹{S0:.2f}")
print(f"ATM Strike: ₹{ATM:.2f}")
print(f"Volatility: {sigma:.4f} ({sigma*100:.2f}%)")
print(f"Risk-free rate: {r:.4f} ({r*100:.2f}%)")

# Calculate option prices
option_data = []
for K in strikes:
    for T, T_days in zip(maturities, maturities_days):
        call_price = black_scholes_call(S0, K, T, r, sigma)
        put_price = black_scholes_put(S0, K, T, r, sigma)
        
        option_data.append({
            'Strike': K,
            'Maturity_Days': T_days,
            'Maturity_Years': T,
            'Call_Price': call_price,
            'Put_Price': put_price
        })

options_df = pd.DataFrame(option_data)

print("\n" + "-"*80)
print("OPTION PRICING TABLE")
print("-"*80)
print(options_df.to_string(index=False))

# ============================================================================
# PART C: GREEKS & VOLATILITY
# ============================================================================
print("\n" + "="*80)
print("PART C: GREEKS CALCULATION")
print("="*80)

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate all Greeks for an option"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)
    
    # Gamma (same for call and put)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Vega (same for call and put, per 1% change in vol)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    # Theta (per day)
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r*T) * norm.cdf(d2)) / 365
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r*T) * norm.cdf(-d2)) / 365
    
    # Rho (per 1% change in rate)
    if option_type == 'call':
        rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100
    
    return {'Delta': delta, 'Gamma': gamma, 'Vega': vega, 'Theta': theta, 'Rho': rho}

# Calculate Greeks for all options
greeks_data = []
for _, row in options_df.iterrows():
    K = row['Strike']
    T = row['Maturity_Years']
    T_days = row['Maturity_Days']
    
    call_greeks = calculate_greeks(S0, K, T, r, sigma, 'call')
    put_greeks = calculate_greeks(S0, K, T, r, sigma, 'put')
    
    greeks_data.append({
        'Strike': K,
        'Maturity': T_days,
        'Type': 'Call',
        'Price': row['Call_Price'],
        **call_greeks
    })
    
    greeks_data.append({
        'Strike': K,
        'Maturity': T_days,
        'Type': 'Put',
        'Price': row['Put_Price'],
        **put_greeks
    })

greeks_df = pd.DataFrame(greeks_data)

print("\n" + "-"*80)
print("GREEKS TABLE (Using Historical Volatility)")
print("-"*80)
print(greeks_df.to_string(index=False))

# Volatility Surface Visualization
print("\n" + "-"*80)
print("VOLATILITY SURFACE")
print("-"*80)

# For demonstration, we'll create a simple IV surface
# In practice, you would calculate IV from market prices
strike_range = np.array(strikes)
maturity_range = np.array(maturities_days)

# Create a grid for volatility surface
X, Y = np.meshgrid(strike_range, maturity_range)
Z = np.ones_like(X) * sigma  # Simplified - constant vol surface

# Add smile effect (demonstration)
for i in range(len(maturity_range)):
    for j in range(len(strike_range)):
        moneyness = strike_range[j] / S0
        # Add volatility smile
        Z[i, j] = sigma * (1 + 0.1 * (moneyness - 1)**2)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('Strike Price')
ax.set_ylabel('Maturity (Days)')
ax.set_zlabel('Implied Volatility')
ax.set_title('Volatility Surface for Maruti Options', fontsize=14, fontweight='bold')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('part_c_volatility_surface.png', dpi=300, bbox_inches='tight')
print("✓ Chart saved: part_c_volatility_surface.png")

# ============================================================================
# PART D: PORTFOLIO & HEDGING
# ============================================================================
print("\n" + "="*80)
print("PART D: PORTFOLIO CONSTRUCTION & HEDGING")
print("="*80)

# Construct a sample portfolio
# Example: Long Call Spread + Short Put
portfolio = [
    {'Strike': ATM * 0.98, 'Maturity': 60, 'Type': 'Call', 'Position': 1},   # Long ITM Call
    {'Strike': ATM * 1.02, 'Maturity': 60, 'Type': 'Call', 'Position': -1},  # Short OTM Call
    {'Strike': ATM * 0.95, 'Maturity': 60, 'Type': 'Put', 'Position': -1},   # Short OTM Put
]

print("\n" + "-"*80)
print("PORTFOLIO COMPOSITION")
print("-"*80)
portfolio_df = pd.DataFrame(portfolio)
print(portfolio_df.to_string(index=False))

# Calculate portfolio Greeks
portfolio_delta = 0
portfolio_gamma = 0
portfolio_vega = 0
portfolio_theta = 0
portfolio_value = 0

for pos in portfolio:
    option_data = greeks_df[
        (greeks_df['Strike'] == pos['Strike']) & 
        (greeks_df['Maturity'] == pos['Maturity']) & 
        (greeks_df['Type'] == pos['Type'])
    ].iloc[0]
    
    portfolio_delta += option_data['Delta'] * pos['Position']
    portfolio_gamma += option_data['Gamma'] * pos['Position']
    portfolio_vega += option_data['Vega'] * pos['Position']
    portfolio_theta += option_data['Theta'] * pos['Position']
    portfolio_value += option_data['Price'] * pos['Position']

print("\n" + "-"*80)
print("UNHEDGED PORTFOLIO GREEKS")
print("-"*80)
unhedged_greeks = pd.DataFrame({
    'Metric': ['Portfolio Value', 'Delta', 'Gamma', 'Vega', 'Theta'],
    'Value': [portfolio_value, portfolio_delta, portfolio_gamma, portfolio_vega, portfolio_theta]
})
print(unhedged_greeks.to_string(index=False))

# Delta Hedging
hedge_shares = -portfolio_delta
hedged_delta = portfolio_delta + hedge_shares * 1  # Delta of stock is 1

print("\n" + "-"*80)
print("DELTA HEDGING")
print("-"*80)
print(f"Shares to hedge: {hedge_shares:.4f}")
print(f"Hedged Delta: {hedged_delta:.6f}")

# Gamma Hedging (using an additional option)
# Use ATM call with 60-day maturity for gamma hedging
gamma_hedge_option = greeks_df[
    (greeks_df['Strike'] == ATM) & 
    (greeks_df['Maturity'] == 60) & 
    (greeks_df['Type'] == 'Call')
].iloc[0]

gamma_hedge_quantity = -portfolio_gamma / gamma_hedge_option['Gamma']
hedged_gamma = portfolio_gamma + gamma_hedge_quantity * gamma_hedge_option['Gamma']
hedged_delta_after_gamma = portfolio_delta + gamma_hedge_quantity * gamma_hedge_option['Delta']

# Adjust stock hedge after gamma hedge
final_stock_hedge = -hedged_delta_after_gamma

print("\n" + "-"*80)
print("GAMMA HEDGING")
print("-"*80)
print(f"Gamma hedge option: {gamma_hedge_option['Type']} Strike={gamma_hedge_option['Strike']}")
print(f"Quantity: {gamma_hedge_quantity:.4f}")
print(f"Hedged Gamma: {hedged_gamma:.6f}")
print(f"Final stock hedge: {final_stock_hedge:.4f}")

# PnL Simulation
price_changes = [-0.02, -0.01, 0, 0.01, 0.02]
pnl_data = []

for pct_change in price_changes:
    S_new = S0 * (1 + pct_change)
    
    # Unhedged PnL
    unhedged_pnl = 0
    for pos in portfolio:
        T = pos['Maturity'] / 365
        K = pos['Strike']
        if pos['Type'] == 'Call':
            new_price = black_scholes_call(S_new, K, T, r, sigma)
            old_price = black_scholes_call(S0, K, T, r, sigma)
        else:
            new_price = black_scholes_put(S_new, K, T, r, sigma)
            old_price = black_scholes_put(S0, K, T, r, sigma)
        
        unhedged_pnl += (new_price - old_price) * pos['Position']
    
    # Hedged PnL (with stock hedge)
    stock_pnl = hedge_shares * (S_new - S0)
    hedged_pnl = unhedged_pnl + stock_pnl
    
    pnl_data.append({
        'Price_Change_%': pct_change * 100,
        'New_Price': S_new,
        'Unhedged_PnL': unhedged_pnl,
        'Stock_Hedge_PnL': stock_pnl,
        'Hedged_PnL': hedged_pnl
    })

pnl_df = pd.DataFrame(pnl_data)

print("\n" + "-"*80)
print("PnL SIMULATION")
print("-"*80)
print(pnl_df.to_string(index=False))

# Plot PnL comparison
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(pnl_df['Price_Change_%'], pnl_df['Unhedged_PnL'], 
        marker='o', linewidth=2, label='Unhedged PnL')
ax.plot(pnl_df['Price_Change_%'], pnl_df['Hedged_PnL'], 
        marker='s', linewidth=2, label='Hedged PnL')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel('Stock Price Change (%)')
ax.set_ylabel('P&L (INR)')
ax.set_title('Hedged vs Unhedged Portfolio P&L', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('part_d_hedging_pnl.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: part_d_hedging_pnl.png")

# ============================================================================
# PART E: VALUE-AT-RISK (VaR)
# ============================================================================
print("\n" + "="*80)
print("PART E: VALUE-AT-RISK CALCULATION")
print("="*80)

# Use last 60 days for VaR calculation
returns_60d = daily_returns.tail(60)

# 1. Parametric VaR
confidence_levels = [0.95, 0.99]
var_results = []

mean_return = returns_60d.mean()
std_return = returns_60d.std()

for conf in confidence_levels:
    z_score = norm.ppf(conf)
    
    # Unhedged VaR (portfolio value is exposed to delta)
    portfolio_std = abs(portfolio_delta) * S0 * std_return
    var_parametric_unhedged = z_score * portfolio_std
    
    # Hedged VaR (much lower due to hedging)
    hedged_std = abs(hedged_delta) * S0 * std_return
    var_parametric_hedged = z_score * hedged_std
    
    # Historical VaR
    unhedged_returns = returns_60d * portfolio_delta * S0
    var_historical_unhedged = np.percentile(unhedged_returns, (1-conf)*100)
    
    hedged_returns = returns_60d * hedged_delta * S0
    var_historical_hedged = np.percentile(hedged_returns, (1-conf)*100)
    
    var_results.append({
        'Confidence_Level': f'{conf*100}%',
        'Parametric_VaR_Unhedged': abs(var_parametric_unhedged),
        'Parametric_VaR_Hedged': abs(var_parametric_hedged),
        'Historical_VaR_Unhedged': abs(var_historical_unhedged),
        'Historical_VaR_Hedged': abs(var_historical_hedged)
    })

var_df = pd.DataFrame(var_results)

print("\n" + "-"*80)
print("VALUE-AT-RISK (1-DAY)")
print("-"*80)
print(var_df.to_string(index=False))

print("\n" + "-"*80)
print("VaR INTERPRETATION")
print("-"*80)
print("Parametric VaR: Based on normal distribution assumption")
print("Historical VaR: Based on actual historical returns distribution")
print("\nKey Observations:")
print(f"• Hedging reduces VaR significantly (by ~{(1-var_df.iloc[0,2]/var_df.iloc[0,1])*100:.1f}%)")
print("• Higher confidence level (99%) shows larger potential losses")
print("• Historical VaR may differ from parametric due to non-normal returns")

# ============================================================================
# SUMMARY & EXPORT
# ============================================================================
print("\n" + "="*80)
print("ASSIGNMENT COMPLETE - SUMMARY")
print("="*80)
print("\n✓ Part A: Data collection and statistics completed")
print("✓ Part B: Option pricing (30 options) completed")
print("✓ Part C: Greeks and volatility surface completed")
print("✓ Part D: Portfolio hedging demonstrated")
print("✓ Part E: VaR analysis completed")

# Export to Excel
with pd.ExcelWriter('FRAM_Maruti_Assignment_Results.xlsx', engine='openpyxl') as writer:
    summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    options_df.to_excel(writer, sheet_name='Option_Prices', index=False)
    greeks_df.to_excel(writer, sheet_name='Greeks', index=False)
    portfolio_df.to_excel(writer, sheet_name='Portfolio', index=False)
    unhedged_greeks.to_excel(writer, sheet_name='Portfolio_Greeks', index=False)
    pnl_df.to_excel(writer, sheet_name='PnL_Simulation', index=False)
    var_df.to_excel(writer, sheet_name='VaR_Analysis', index=False)

print("\n✓ Results exported to: FRAM_Maruti_Assignment_Results.xlsx")
print("\n" + "="*80)
print("END OF ANALYSIS")
print("="*80)