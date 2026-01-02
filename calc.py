import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# ----------------------------
# Inputs
# ----------------------------
st.sidebar.header("Retirement Inputs")
monthly_income = st.sidebar.number_input("Monthly Income Required (₹)", value=50000)
inflation = st.sidebar.number_input("Annual Inflation (%)", value=4.0)
target_xirr = st.sidebar.number_input("Target Equity XIRR (%)", value=12.0)
volatility = st.sidebar.number_input("Equity Volatility (%)", value=15.0)
equity_pct = st.sidebar.number_input("Equity Allocation (%)", value=70)
arbitrage_return = st.sidebar.number_input("Arbitrage Fund Return (% p.a.)", value=7.0)
horizon_years = st.sidebar.number_input("Time Horizon (Years)", value=40)
starting_age = st.sidebar.number_input("Starting Age", value=60)
target_conf = st.sidebar.number_input("Target Confidence Level (%)", value=90)
harvest_threshold = st.sidebar.number_input("Harvest Threshold (%)", value=10.0)
harvest_rate = st.sidebar.number_input("Harvest Rate (%)", value=5.0)

# ----------------------------
# Monte Carlo Simulation Function
# ----------------------------
def run_simulation(initial_capital, n_sims=5):
    horizon_months = horizon_years * 12
    equity_allocation = equity_pct / 100
    arb_allocation = 1 - equity_allocation
    inflation_monthly = (1 + inflation/100)**(1/12) - 1
    arb_monthly = (1 + arbitrage_return/100)**(1/12) - 1
    log_monthly_target = np.log(1 + target_xirr/100)/12
    monthly_vol = (volatility/100)/np.sqrt(12)
    harvest_threshold_dec = harvest_threshold/100
    harvest_rate_dec = harvest_rate/100
    
    # Store results
    all_paths = []
    for sim in range(n_sims):
        equity = initial_capital * equity_allocation
        arb = initial_capital * arb_allocation
        monthly_withdrawal = monthly_income
        annual_tracker = 1.0
        ages = []
        total_corpus = []
        equity_list = []
        arb_list = []
        
        for month in range(1, horizon_months+1):
            # Equity log-normal return
            z = np.random.normal()
            m_ret = np.exp(log_monthly_target + monthly_vol * z) - 1
            equity *= (1 + m_ret)
            annual_tracker *= (1 + m_ret)
            
            # Arbitrage growth
            arb *= (1 + arb_monthly)
            
            # Withdraw
            if arb >= monthly_withdrawal:
                arb -= monthly_withdrawal
            else:
                shortfall = monthly_withdrawal - arb
                arb = 0
                equity -= shortfall
            
            if equity + arb <= 0:
                equity, arb = 0, 0
                break
            
            # Annual adjustments
            if month % 12 == 0:
                monthly_withdrawal *= (1 + inflation/100)
                actual_annual_return = annual_tracker - 1
                if actual_annual_return > harvest_threshold_dec:
                    transfer = equity * harvest_rate_dec
                    equity -= transfer
                    arb += transfer
                annual_tracker = 1.0
            
            # Store yearly data
            if month % 12 == 0:
                age = starting_age + month // 12
                ages.append(age)
                equity_list.append(equity)
                arb_list.append(arb)
                total_corpus.append(equity + arb)
        
        df = pd.DataFrame({
            "age": ages,
            "equity": equity_list,
            "arbitrage": arb_list,
            "total": total_corpus
        })
        all_paths.append(df)
    
    # Calculate percentiles
    combined_total = pd.concat([p[['total']] for p in all_paths], axis=1)
    total_p25 = combined_total.quantile(0.25, axis=1)
    total_p50 = combined_total.quantile(0.5, axis=1)
    total_p75 = combined_total.quantile(0.75, axis=1)
    
    return all_paths, total_p25, total_p50, total_p75, ages

# ----------------------------
# Run Simulation
# ----------------------------
initial_capital = monthly_income * 100  # initial guess
paths, p25, p50, p75, ages = run_simulation(initial_capital, n_sims=5)

# ----------------------------
# Plotting with Plotly
# ----------------------------
fig = go.Figure()

# Individual paths
colors = ['lightgray']*5
for i, df in enumerate(paths):
    fig.add_trace(go.Scatter(x=df['age'], y=df['total'], mode='lines', name=f'Path {i+1}', line=dict(color=colors[i], width=1)))

# Percentile shading
fig.add_trace(go.Scatter(x=ages + ages[::-1],
                         y=list(p75) + list(p25[::-1]),
                         fill='toself',
                         fillcolor='rgba(0,100,80,0.2)',
                         line=dict(color='rgba(255,255,255,0)'),
                         hoverinfo="skip",
                         showlegend=True,
                         name='25th-75th Percentile'))

# Median line
fig.add_trace(go.Scatter(x=ages, y=p50, mode='lines', name='Median (P50)', line=dict(color='purple', width=3)))

fig.update_layout(title="Portfolio Corpus Evolution Over Time",
                  xaxis_title="Age (Years)",
                  yaxis_title="Total Corpus (₹)",
                  template="plotly_white",
                  height=400)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Display Summary
# ----------------------------
starting_total = initial_capital
median_start = p50.iloc[0]
p75_end = p75.iloc[-1]
p25_end = p25.iloc[-1]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Starting Total", f"₹{starting_total/1e7:.2f} Cr")
col2.metric(f"Median at Age {starting_age}", f"₹{median_start/1e7:.2f} Cr")
col3.metric("75th %ile at End", f"₹{p75_end/1e7:.2f} Cr")
col4.metric("25th %ile at End", f"₹{p25_end/1e7:.2f} Cr")
