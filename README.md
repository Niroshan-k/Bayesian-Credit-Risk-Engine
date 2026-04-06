# Bayesian Credit Risk Engine

> A probabilistic ML tool for banks and financial institutions — predicts loan default probability with confidence, not just a number.

![Dashboard](data/dashboard.png)

---

## Mathematical Concept Breakdown

| | | |
|:---:|:---:|:---:|
| ![](images/1.png) | ![](images/2.png) | ![](images/3.png) |
| ![](images/4.png) | ![](images/5.png) | ![](images/6.png) |
| ![](images/7.png) | ![](images/8.png) | ![](images/9.png) |
| ![](images/10.png) | ![](images/11.png) | ![](images/12.png) |
| | | |

---

## What Is This?

Most ML models give you a single prediction — 84% chance of default. That number has no uncertainty attached to it. The model is completely confident, whether it should be or not.

In banking, that's a problem. Knowing **how confident** the model is matters just as much as the prediction itself.

This engine uses **Bayesian inference** to produce a full probability distribution instead of a single number. For each loan applicant it returns:

- A **mean default probability** — the best estimate
- A **confidence range** — how certain the model is
- A **dynamic interest rate** — priced according to the actual risk distribution

If the model is uncertain about a borderline case, it flags it for human review rather than making a blind automated decision.

---

## Why Bayesian?

Standard ML finds one set of weights and commits to them. Bayesian ML treats weights as distributions — reflecting the uncertainty about what the true values really are.

Under the hood this means solving:

```
P(β | data) = P(data | β) × P(β) / P(data)
```

The denominator `P(data)` requires integrating over every possible combination of parameters — which becomes computationally impossible as the number of features grows. **MCMC (Markov Chain Monte Carlo)** solves this by sampling the posterior distribution without ever computing that integral directly.

The result is a matrix of thousands of plausible weight combinations. For each new applicant, the model runs through every row, producing a distribution of predictions rather than a single output.

---

## Key Features

- **Probabilistic predictions** — full posterior distribution per applicant, not a point estimate
- **Uncertainty quantification** — std and 95% confidence interval on every prediction
- **Dynamic interest rate pricing** — rates assigned based on risk distribution thresholds
- **Explainability** — parameter distributions show which features drive default risk
- **Interactive dashboard** — real-time Streamlit UI for live applicant evaluation

---

## Relevance to Banking Regulation

- **IFRS 9** — requires Expected Credit Loss estimation with probability-weighted scenarios. This model naturally produces those distributions.
- **Basel III IRB** — requires Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). Uncertainty quantification strengthens these estimates.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Core ML | PyMC, ArviZ, scikit-learn |
| Data | NumPy, Pandas |
| Dashboard | Streamlit, Matplotlib |

---

## Project Structure

```
├── main.py          # trains the model, runs MCMC, exports weights
├── dashboard.py     # Streamlit frontend, loads model, evaluates applicants
├── src/             # helper modules for data fetching and cleaning
├── model/           # compiled NetCDF model + scaler files (git ignored)
├── data/            # raw datasets and assets
└── images/          # concept breakdown slides
```

---

## How to Run

**1. Install dependencies**
```bash
pip install pymc arviz streamlit scikit-learn pandas numpy matplotlib joblib kagglehub
```

**2. Train the model**
```bash
python main.py
```

**3. Launch the dashboard**
```bash
streamlit run dashboard.py
```

---

## Author

**Lakshan Niroshan** — [GitHub](https://github.com/Niroshan-k) · [LinkedIn](https://www.linkedin.com/in/niroshank)