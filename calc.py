import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Retirement Capital Adequacy Calculator")

st.title("Retirement Capital Adequacy Calculator")

# --- Inputs ---
st.sidebar.header("User Inputs")

monthly_income = st.sidebar.number_input("Monthly Income Required (₹)", value=50000, step=5000)
inflation = st.sidebar.number_input("Annual Inflation (%)", value=4.0, step=0.5)
target_xirr = st.sidebar.number_input("Target Equity XIRR (%)", value=12.0, step=0.5)
volatility = st.sidebar.number_input("Equity Volatility (%)", value=15.0, step=0.5)
equity_pct = st.sidebar.number_input("Equity Allocation (%)", value=70, step=5)
arbitrage_return = st.sidebar.number_input("Arbitrage Fund Return (% p.a.)", value=7.0, step=0.5)
horizon_years = st.sidebar.number_input("Time Horizon (Years)", value=40, step=1)
starting_age = st.sidebar.number_input("Starting Age", value=60, step=1)
target_confidence = st.sidebar.number_input("Target Confidence Level (%)", value=90, step=1)
harvest_threshold = st.sidebar.number_input("Harvest Threshold (%)", value=10.0, step=1)
harvest_rate = st.sidebar.number_input("Harvest Rate (%)", value=5.0, step=1)

# --- Monte Carlo Simulation Function ---
def run_simulation(initial_capital, params, num_sims=2000, track_income=False, track_corpus=False):
    monthly_income, inflation, target_xirr, volatility, equity_pct, arbitrage_return, horizon_years, starting_age, harvest_threshold, harvest_rate = params
    inflation_annual = inflation / 100
    target_xirr_annual = target_xirr / 100
    volatility_annual = volatility / 100
    equity_allocation = equity_pct / 100
    arbitrage_return_annual = arbitrage_return / 100
    arbitrage_return_monthly = (1 + arbitrage_return_annual)**(1/12) - 1
    horizon_months = horizon_years * 12

    log_monthly_target = np.log(1 + target_xirr_annual) / 12
    monthly_vol = volatility_annual / np.sqrt(12)
    harvest_threshold_decimal = harvest_threshold / 100
    harvest_rate_decimal = harvest_rate / 100

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
            sim_income_tracker.append({'year':0, 'age':starting_age, 'monthly_income':current_withdrawal, 'annual_income':current_withdrawal*12})
        if track_corpus:
            sim_corpus_tracker.append({'year':0, 'age':starting_age, 'equity':equity, 'arbitrage':arbitrage, 'total':equity+arbitrage})

        for month in range(1, horizon_months + 1):
            # Log-normal equity return
            normal_random = np.random.normal()
            m_ret = np.exp(log_monthly_target + monthly_vol * normal_random) - 1
            equity *= (1 + m_ret)
            annual_equity_tracker *= (1 + m_ret)

            # Arbitrage growth
            arbitrage *= (1 + arbitrage_return_monthly)

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

            # Annual updates
            if month % 12 == 0:
                current_withdrawal *= (1 + inflation_annual)
                if track_income:
                    sim_income_tracker.append({'year':month//12, 'age':starting_age+month//12, 'monthly_income':current_withdrawal, 'annual_income':current_withdrawal*12})
                if track_corpus:
                    sim_corpus_tracker.append({'year':month//12, 'age':starting_age+month//12, 'equity':equity, 'arbitrage':arbitrage, 'total':equity+arbitrage})

                # Harvest
                actual_annual_return = annual_equity_tracker - 1
                if actual_annual_return > harvest_threshold_decimal:
                    transfer_amt = equity * harvest_rate_decimal
                    equity -= transfer_amt
                    arbitrage += transfer_amt
                annual_equity_tracker = 1.0

        if not portfolio_failed:
            success_count += 1
            terminal_values.append(equity + arbitrage)
            if track_income:
                income_projections.append(sim_income_tracker)
            if track_corpus:
                corpus_projections.append(sim_corpus_tracker)
        else:
            terminal_values.append(0)

    confidence_level = (success_count / num_sims) * 100
    successful_values = sorted([v for v in terminal_values if v > 0])
    p25 = successful_values[int(len(successful_values)*0.25)] if successful_values else 0
    p50 = successful_values[int(len(successful_values)*0.50)] if successful_values else 0
    p75 = successful_values[int(len(successful_values)*0.75)] if successful_values else 0

    return {
        'confidence_level': confidence_level,
        'terminal_values': {'p25':p25, 'p50':p50, 'p75':p75},
        'income_projections': income_projections,
        'corpus_projections': corpus_projections
    }

# --- Capital Adequacy Search ---
if st.button("Run Simulation"):
    st.info("Running Monte Carlo simulation... this may take a few seconds.")

    params = (monthly_income, inflation, target_xirr, volatility, equity_pct, arbitrage_return, horizon_years, starting_age, harvest_threshold, harvest_rate)

    # Search range for required capital
    min_capital = monthly_income * 100
    max_capital = monthly_income * 400
    steps = 15
    capital_range = []

    for i in range(steps+1):
        capital = min_capital + (max_capital - min_capital) * (i / steps)
        sim_result = run_simulation(capital, params, num_sims=500)
        capital_range.append({'Capital':capital/1e7, 'Confidence':sim_result['confidence_level'], 'TerminalP50':sim_result['terminal_values']['p50']/1e7})

    curve_df = pd.DataFrame(capital_range)

    # Interpolate required capital
    req_capital = None
    for i in range(len(curve_df)-1):
        curr = curve_df.iloc[i]
        next_ = curve_df.iloc[i+1]
        if curr['Confidence'] <= target_confidence <= next_['Confidence']:
            ratio = (target_confidence - curr['Confidence']) / (next_['Confidence'] - curr['Confidence'])
            req_capital = curr['Capital'] + ratio * (next_['Capital'] - curr['Capital'])
            break

    st.success(f"Required Capital: ₹ {req_capital:.2f} Crores" if req_capital else "Required capital not found in range.")

    # Detailed simulation at required capital
    if req_capital:
        detailed = run_simulation(req_capital*1e7, params, num_sims=1000, track_income=True, track_corpus=True)

        # Process data for plots
        years = np.arange(0, horizon_years+1)
        # Median total
        total_median = [np.median([sim[y]['total']/1e7 for sim in detailed['corpus_projections'] if len(sim)>y]) for y in years]
        equity_median = [np.median([sim[y]['equity']/1e7 for sim in detailed['corpus_projections'] if len(sim)>y]) for y in years]
        arb_median = [np.median([sim[y]['arbitrage']/1e7 for sim in detailed['corpus_projections'] if len(sim)>y]) for y in years]
        # Income
        monthly_income_series = [np.mean([sim[y]['monthly_income'] for sim in detailed['income_projections'] if len(sim)>y]) for y in years]

        FIG_W, FIG_H = 5, 3.5

        # --- Layout: 2x2 Grid ---
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        # Plot 1: Confidence vs Capital
        with col1:
            fig1, ax1 = plt.subplots(figsize=(FIG_W, FIG_H))
            ax1.plot(curve_df["Capital"], curve_df["Confidence"], marker='o')
            ax1.axhline(target_confidence, color='r', linestyle='--', label=f"{target_confidence}% Target")
            ax1.set_xlabel("Capital (₹ Cr)")
            ax1.set_ylabel("Confidence (%)")
            ax1.set_title("Confidence vs Capital")
            ax1.legend()
            plt.tight_layout()
            st.pyplot(fig1)

        # Plot 2: Portfolio Total Corpus
        with col2:
            fig2, ax2 = plt.subplots(figsize=(FIG_W, FIG_H))
            ax2.plot(starting_age + years, total_median, color='purple', linewidth=2, label='Median Total Corpus')
            ax2.set_xlabel("Age")
            ax2.set_ylabel("Corpus (₹ Cr)")
            ax2.set_title("Median Portfolio Total Corpus")
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig2)

        # Plot 3: Equity vs Arbitrage
        with col3:
            fig3, ax3 = plt.subplots(figsize=(FIG_W, FIG_H))
            ax3.plot(starting_age + years, equity_median, color='blue', label='Equity (Median)')
            ax3.plot(starting_age + years, arb_median, color='green', label='Arbitrage (Median)')
            ax3.set_xlabel("Age")
            ax3.set_ylabel("Corpus (₹ Cr)")
            ax3.set_title("Equity vs Arbitrage Corpus")
            ax3.legend()
            plt.tight_layout()
            st.pyplot(fig3)

        # Plot 4: Monthly Income Projection
        with col4:
            fig4, ax4 = plt.subplots(figsize=(FIG_W, FIG_H))
            ax4.plot(starting_age + years, monthly_income_series, color='teal', label='Monthly Income')
            ax4.set_xlabel("Age")
            ax4.set_ylabel("Monthly Income (₹)")
            ax4.set_title("Inflation-Adjusted Monthly Income")
            ax4.legend()
            plt.tight_layout()
            st.pyplot(fig4)
