import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Retirement Capital Adequacy Calculator")

st.title("Retirement Capital Adequacy Calculator")

# -------------------------
# Sidebar Inputs
# -------------------------
monthly_income = st.sidebar.number_input("Monthly Income Required (₹)", value=50000, step=1000)
inflation = st.sidebar.number_input("Annual Inflation (%)", value=4.0, step=0.1)
target_xirr = st.sidebar.number_input("Target Equity XIRR (%)", value=12.0, step=0.1)
volatility = st.sidebar.number_input("Equity Volatility (%)", value=15.0, step=0.1)
equity_pct = st.sidebar.number_input("Equity Allocation (%)", value=70, step=1)
arbitrage_return = st.sidebar.number_input("Arbitrage Fund Return (% p.a.)", value=7.0, step=0.1)
horizon_years = st.sidebar.number_input("Time Horizon (Years)", value=40, step=1)
starting_age = st.sidebar.number_input("Starting Age", value=60, step=1)
target_confidence = st.sidebar.number_input("Target Confidence Level (%)", value=90, step=1)
harvest_threshold = st.sidebar.number_input("Harvest Threshold (%)", value=10.0, step=0.5)
harvest_rate = st.sidebar.number_input("Harvest Rate (%)", value=5.0, step=0.5)

# -------------------------
# Monte Carlo Simulation Engine
# -------------------------
def run_simulation(initial_capital, params, num_sims=2000, track_income=False, track_corpus=False):
    monthly_income, inflation, target_xirr, volatility, equity_pct, arbitrage_return, horizon_years, starting_age, harvest_threshold, harvest_rate = params
    
    inflation /= 100
    target_xirr /= 100
    volatility /= 100
    equity_allocation = equity_pct / 100
    arbitrage_return /= 100
    arbitrage_monthly = (1 + arbitrage_return)**(1/12) - 1
    horizon_months = int(horizon_years * 12)
    log_monthly_target = np.log(1 + target_xirr)/12
    monthly_vol = volatility/np.sqrt(12)
    harvest_threshold /= 100
    harvest_rate /= 100
    
    success_count = 0
    terminal_values = []
    income_projections = []
    corpus_projections = []

    for sim in range(num_sims):
        equity = initial_capital * equity_allocation
        arbitrage = initial_capital * (1 - equity_allocation)
        current_withdrawal = monthly_income
        portfolio_failed = False
        annual_equity_tracker = 1.0

        sim_income_tracker = []
        sim_corpus_tracker = []

        if track_income:
            sim_income_tracker.append({'year':0,'age':starting_age,'monthly_income':current_withdrawal,'annual_income':current_withdrawal*12})
        if track_corpus:
            sim_corpus_tracker.append({'year':0,'age':starting_age,'equity':equity,'arbitrage':arbitrage,'total':equity+arbitrage})

        for month in range(1,horizon_months+1):
            norm_rand = np.random.normal()
            mret = np.exp(log_monthly_target + monthly_vol*norm_rand) - 1
            equity *= (1 + mret)
            annual_equity_tracker *= (1 + mret)
            arbitrage *= (1 + arbitrage_monthly)

            # Withdrawal
            if arbitrage >= current_withdrawal:
                arbitrage -= current_withdrawal
            else:
                shortfall = current_withdrawal - arbitrage
                arbitrage = 0
                equity -= shortfall

            if equity + arbitrage <= 0:
                portfolio_failed = True
                break

            # Annual adjustments
            if month % 12 == 0:
                current_withdrawal *= (1 + inflation)
                if track_income:
                    sim_income_tracker.append({'year':month//12,'age':starting_age + month//12,'monthly_income':current_withdrawal,'annual_income':current_withdrawal*12})
                if track_corpus:
                    sim_corpus_tracker.append({'year':month//12,'age':starting_age + month//12,'equity':equity,'arbitrage':arbitrage,'total':equity+arbitrage})
                
                actual_return = annual_equity_tracker - 1
                if actual_return > harvest_threshold:
                    transfer = equity * harvest_rate
                    equity -= transfer
                    arbitrage += transfer
                annual_equity_tracker = 1.0

        if not portfolio_failed:
            success_count += 1
            terminal_values.append(equity + arbitrage)
            if track_income: income_projections.append(sim_income_tracker)
            if track_corpus: corpus_projections.append(sim_corpus_tracker)
        else:
            terminal_values.append(0)
    
    confidence_level = (success_count / num_sims) * 100
    successful_values = sorted([v for v in terminal_values if v>0])
    p25 = successful_values[int(len(successful_values)*0.25)] if successful_values else 0
    p50 = successful_values[int(len(successful_values)*0.5)] if successful_values else 0
    p75 = successful_values[int(len(successful_values)*0.75)] if successful_values else 0

    return {
        'confidence': confidence_level,
        'terminal': {'p25':p25, 'p50':p50, 'p75':p75},
        'income': income_projections,
        'corpus': corpus_projections
    }

# -------------------------
# Calculate Required Capital
# -------------------------
if st.button("Calculate Required Capital"):
    st.info("Running Monte Carlo simulation, please wait...")
    params = (monthly_income, inflation, target_xirr, volatility, equity_pct, arbitrage_return, horizon_years, starting_age, harvest_threshold, harvest_rate)

    # Capital sweep to find required capital
    min_cap = monthly_income*100
    max_cap = monthly_income*400
    steps = 15
    capital_range = []

    for i in range(steps+1):
        cap = min_cap + (max_cap - min_cap)*(i/steps)
        sim = run_simulation(cap, params, num_sims=500)
        capital_range.append({'capital':cap/1e7,'confidence':sim['confidence']})

    # Interpolation for required capital
    req_cap = None
    for i in range(len(capital_range)-1):
        curr, nxt = capital_range[i], capital_range[i+1]
        if curr['confidence'] <= target_confidence <= nxt['confidence']:
            ratio = (target_confidence - curr['confidence'])/(nxt['confidence'] - curr['confidence'])
            req_cap = curr['capital'] + ratio*(nxt['capital'] - curr['capital'])
            break

    st.success(f"Required Capital: ₹ {req_cap:.2f} Crores" if req_cap else "Increase range or adjust parameters")

    # Detailed simulation at required capital
    if req_cap:
        det = run_simulation(req_cap*1e7, params, num_sims=1000, track_income=True, track_corpus=True)

        # Prepare DataFrames for plots
        years = list(range(horizon_years+1))

        # Income chart
        income_avg = []
        for yr in years:
            vals = [sim[yr]['monthly_income'] for sim in det['income'] if len(sim)>yr]
            income_avg.append(np.mean(vals) if vals else 0)
        df_income = pd.DataFrame({'Year':years, 'Age':[starting_age + y for y in years], 'Monthly Income':income_avg})

        # Corpus chart - median + percentiles
        corpus_data = {'Year':years, 'Age':[starting_age + y for y in years]}
        for key in ['p25','p50','p75']:
            corpus_data[f'Total_{key}'] = []
        for yr in years:
            totals = [sim[yr]['total']/1e7 for sim in det['corpus'] if len(sim)>yr]
            totals.sort()
            n = len(totals)
            corpus_data['Total_p25'].append(totals[int(0.25*n)] if n>0 else 0)
            corpus_data['Total_p50'].append(totals[int(0.5*n)] if n>0 else 0)
            corpus_data['Total_p75'].append(totals[int(0.75*n)] if n>0 else 0)
        df_corpus = pd.DataFrame(corpus_data)

        # Capital vs confidence chart
        df_conf = pd.DataFrame(capital_range)

        # -------------------------
        # Plotting with Plotly in 2x2 grid
        # -------------------------
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_corpus['Age'], y=df_corpus['Total_p50'], mode='lines', name='Median', line=dict(color='purple', width=3)))
        fig1.add_trace(go.Scatter(x=df_corpus['Age'], y=df_corpus['Total_p25'], mode='lines', name='25th %ile', line=dict(color='red', dash='dash')))
        fig1.add_trace(go.Scatter(x=df_corpus['Age'], y=df_corpus['Total_p75'], mode='lines', name='75th %ile', line=dict(color='green', dash='dash')))
        fig1.update_layout(title='Portfolio Total Corpus Over Time', xaxis_title='Age', yaxis_title='Total Corpus (₹ Crores)', height=350)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_income['Age'], y=df_income['Monthly Income'], mode='lines', name='Monthly Income', line=dict(color='green', width=3)))
        fig2.update_layout(title='Monthly Income Projection', xaxis_title='Age', yaxis_title='Monthly Income (₹)', height=350)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df_corpus['Age'], y=df_corpus['Total_p50'], mode='lines', name='Median Total', line=dict(color='purple', width=3)))
        fig3.add_trace(go.Scatter(x=df_corpus['Age'], y=[det['corpus'][0][yr]['equity']/1e7 if len(det['corpus'][0])>yr else 0 for yr in years], mode='lines', name='Equity (Example Path)', line=dict(color='blue', dash='dot')))
        fig3.add_trace(go.Scatter(x=df_corpus['Age'], y=[det['corpus'][0][yr]['arbitrage']/1e7 if len(det['corpus'][0])>yr else 0 for yr in years], mode='lines', name='Arbitrage (Example Path)', line=dict(color='green', dash='dot')))
        fig3.update_layout(title='Equity vs Arbitrage Corpus (Example Path)', xaxis_title='Age', yaxis_title='Corpus (₹ Crores)', height=350)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=df_conf['capital'], y=df_conf['confidence'], mode='lines+markers', name='Confidence', line=dict(color='blue', width=2)))
        fig4.add_trace(go.Scatter(x=[req_cap], y=[target_confidence], mode='markers', name='Target', marker=dict(color='red', size=10)))
        fig4.update_layout(title='Capital vs Confidence Curve', xaxis_title='Initial Capital (₹ Crores)', yaxis_title='Confidence (%)', height=350)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(fig3, use_container_width=True)
        with col4:
            st.plotly_chart(fig4, use_container_width=True)
