import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

st.set_page_config(layout="wide", page_title="Retirement Capital Adequacy Calculator")

# ============================================================
# MONTE CARLO ENGINE (FULL PORT)
# ============================================================
def run_simulation(initial_capital, p, sims=2000, track=True):
    mi = p["monthly_income"]
    infl = p["inflation"] / 100
    xirr = p["xirr"] / 100
    vol = p["vol"] / 100
    eq_pct = p["equity_pct"] / 100
    arb_ret = p["arb_ret"] / 100
    years = p["years"]
    age0 = p["age"]
    ht = p["harvest_threshold"] / 100
    hr = p["harvest_rate"] / 100

    months = years * 12
    arb_m = (1 + arb_ret) ** (1 / 12) - 1

    log_mu = np.log(1 + xirr) / 12
    sigma = vol / np.sqrt(12)

    successes = 0
    terminals = []
    income_paths = []
    corpus_paths = []

    for _ in range(sims):
        equity = initial_capital * eq_pct
        arb = initial_capital * (1 - eq_pct)
        wd = mi
        annual_tracker = 1.0

        inc_path = []
        cor_path = []

        failed = False

        for m in range(1, months + 1):
            z = np.random.normal()
            r = np.exp(log_mu + sigma * z) - 1

            equity *= (1 + r)
            annual_tracker *= (1 + r)
            arb *= (1 + arb_m)

            if arb >= wd:
                arb -= wd
            else:
                equity -= (wd - arb)
                arb = 0

            if equity + arb <= 0:
                failed = True
                break

            if m % 12 == 0:
                wd *= (1 + infl)
                y = m // 12

                if annual_tracker - 1 > ht:
                    tr = equity * hr
                    equity -= tr
                    arb += tr

                inc_path.append({
                    "year": y,
                    "age": age0 + y,
                    "monthly": wd
                })

                cor_path.append({
                    "year": y,
                    "age": age0 + y,
                    "equity": equity,
                    "arb": arb,
                    "total": equity + arb
                })

                annual_tracker = 1.0

        if not failed:
            successes += 1
            terminals.append(equity + arb)
            income_paths.append(inc_path)
            corpus_paths.append(cor_path)
        else:
            terminals.append(0)

    terminals = np.array(terminals)
    valid = terminals[terminals > 0]

    return {
        "confidence": successes / sims * 100,
        "p25": np.percentile(valid, 25) if len(valid) else 0,
        "p50": np.percentile(valid, 50) if len(valid) else 0,
        "p75": np.percentile(valid, 75) if len(valid) else 0,
        "income": income_paths,
        "corpus": corpus_paths
    }

# ============================================================
# SIDEBAR INPUTS
# ============================================================
st.sidebar.header("Inputs")

params = {
    "monthly_income": st.sidebar.number_input("Monthly Income (₹)", 10000, 500000, 50000, 5000),
    "inflation": st.sidebar.number_input("Inflation (%)", 0.0, 8.0, 4.0, 0.1),
    "xirr": st.sidebar.number_input("Equity XIRR (%)", 6.0, 15.0, 12.0, 0.1),
    "vol": st.sidebar.number_input("Equity Volatility (%)", 5.0, 30.0, 15.0, 0.5),
    "equity_pct": st.sidebar.slider("Equity Allocation (%)", 0, 100, 70),
    "arb_ret": st.sidebar.number_input("Arbitrage Return (% p.a.)", 4.0, 9.0, 7.0, 0.1),
    "harvest_threshold": st.sidebar.number_input("Harvest Threshold (%)", 5.0, 15.0, 10.0, 0.5),
    "harvest_rate": st.sidebar.number_input("Harvest Rate (%)", 2.0, 15.0, 5.0, 0.5),
    "years": st.sidebar.number_input("Horizon (Years)", 20, 60, 40),
    "age": st.sidebar.number_input("Starting Age", 40, 70, 60),
    "target_conf": st.sidebar.slider("Target Confidence (%)", 70, 99, 90)
}

# ============================================================
# MAIN CALCULATION
# ============================================================
st.title("Retirement Capital Adequacy Calculator")

if st.button("Run Monte Carlo Simulation"):
    with st.spinner("Running simulations..."):

        caps = np.linspace(
            params["monthly_income"] * 100,
            params["monthly_income"] * 400,
            15
        )

        curve = []
        for c in caps:
            res = run_simulation(c, params, sims=1000, track=False)
            curve.append((c / 1e7, res["confidence"]))

        curve_df = pd.DataFrame(curve, columns=["Capital (Cr)", "Confidence"])

        # Interpolate required capital
        req = None
        for i in range(len(curve_df) - 1):
            if curve_df.iloc[i]["Confidence"] <= params["target_conf"] <= curve_df.iloc[i+1]["Confidence"]:
                x1, y1 = curve_df.iloc[i]
                x2, y2 = curve_df.iloc[i+1]
                req = x1 + (params["target_conf"] - y1) * (x2 - x1) / (y2 - y1)
                break

        st.subheader("Required Capital")
        st.success(f"₹ {req:.2f} Crores")

        detail = run_simulation(req * 1e7, params, sims=2000)

# ============================================================
# GRAPH 1: CAPITAL VS CONFIDENCE
# ============================================================
        st.subheader("Capital vs Confidence")
        fig, ax = plt.subplots()
        ax.plot(curve_df["Capital (Cr)"], curve_df["Confidence"], marker="o")
        ax.axhline(params["target_conf"], linestyle="--")
        ax.set_xlabel("Capital (₹ Cr)")
        ax.set_ylabel("Confidence (%)")
        st.pyplot(fig)

# ============================================================
# GRAPH 2: CORPUS EVOLUTION (PATHS + PERCENTILES)
# ============================================================
        years = list(range(1, params["years"] + 1))
        totals = {y: [] for y in years}

        for sim in detail["corpus"]:
            for r in sim:
                totals[r["year"]].append(r["total"] / 1e7)

        p25 = [np.percentile(totals[y], 25) for y in years]
        p50 = [np.percentile(totals[y], 50) for y in years]
        p75 = [np.percentile(totals[y], 75) for y in years]

        fig, ax = plt.subplots(figsize=(10, 5))

        sample = random.sample(detail["corpus"], 5)
        for s in sample:
            ax.plot([r["year"] for r in s], [r["total"] / 1e7 for r in s], color="gray", alpha=0.4)

        ax.plot(years, p50, label="Median", linewidth=3)
        ax.plot(years, p25, linestyle="--", label="25th %ile")
        ax.plot(years, p75, linestyle="--", label="75th %ile")
        ax.set_xlabel("Years")
        ax.set_ylabel("Corpus (₹ Cr)")
        ax.legend()
        st.pyplot(fig)

# ============================================================
# GRAPH 3: EQUITY VS ARBITRAGE (MEDIAN)
# ============================================================
        eq_med = []
        arb_med = []

        for y in years:
            eq_vals, arb_vals = [], []
            for sim in detail["corpus"]:
                for r in sim:
                    if r["year"] == y:
                        eq_vals.append(r["equity"] / 1e7)
                        arb_vals.append(r["arb"] / 1e7)
            eq_med.append(np.percentile(eq_vals, 50))
            arb_med.append(np.percentile(arb_vals, 50))

        fig, ax = plt.subplots()
        ax.plot(years, eq_med, label="Equity")
        ax.plot(years, arb_med, label="Arbitrage")
        ax.set_xlabel("Years")
        ax.set_ylabel("Corpus (₹ Cr)")
        ax.legend()
        st.pyplot(fig)

# ============================================================
# GRAPH 4: MONTHLY INCOME PROJECTION
# ============================================================
        income_med = []
        for y in years:
            vals = []
            for sim in detail["income"]:
                for r in sim:
                    if r["year"] == y:
                        vals.append(r["monthly"])
            income_med.append(np.mean(vals))

        fig, ax = plt.subplots()
        ax.plot(years, income_med)
        ax.set_xlabel("Years")
        ax.set_ylabel("Monthly Income (₹)")
        st.pyplot(fig)
