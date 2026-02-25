import streamlit as st
import pandas as pd
import numpy as np
import arviz as az
import joblib
import matplotlib.pyplot as plt

# --- 1. Page Config & Styling ---
st.set_page_config(page_title="Bayesian Credit Risk", layout="wide")
st.title("Bayesian SME Credit Risk Engine")

# --- 2. Load Brain & Translator ---
@st.cache_resource
def load_artifacts():
    trace = az.from_netcdf("model/sme_risk_model.nc")
    scaler = joblib.load("model/scaler.pkl")
    return trace, scaler

trace, scaler = load_artifacts()

# --- 3. Advanced UI Layout ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Loan Details")
    loan_amnt = st.number_input("Loan Amount ($)", 1000, 50000, 10000)
    term = st.selectbox("Term (Months)", [36, 60])
    int_rate = st.number_input("Assigned Interest Rate (%)", 5.0, 30.0, 12.0)
    installment = st.number_input("Monthly Installment ($)", 30.0, 1500.0, 300.0)
    purpose = st.selectbox("Purpose", ["debt_consolidation", "credit_card", "home_improvement", "small_business", "other"])

with col2:
    st.subheader("Borrower Profile")
    annual_inc = st.number_input("Annual Income ($)", 10000, 500000, 60000)
    emp_length = st.slider("Employment Length (Years)", 0, 10, 5)
    grade_letter = st.selectbox("LendingClub Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN"])
    verification_status = st.selectbox("Income Verification", ["Not Verified", "Source Verified", "Verified"])

with col3:
    st.subheader("Credit History")
    dti = st.number_input("Debt-to-Income (DTI) %", 0.0, 50.0, 15.0)
    open_acc = st.number_input("Open Credit Lines", 1, 50, 10)
    total_acc = st.number_input("Total Credit Lines", 1, 100, 20)
    revol_util = st.number_input("Revolving Utilization (%)", 0.0, 150.0, 50.0)
    pub_rec = st.number_input("Public Records (Bankruptcies)", 0, 10, 0)

# --- 4. The Bayesian Math Engine ---
if st.button("Calculate Bayesian Risk", type="primary", use_container_width=True):
    st.divider()
    
    # Map the grade back to numbers
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    
    # Build the dictionary exactly as the clean_data function did
    input_dict = {
        'loan_amnt': float(loan_amnt), 'term': float(term), 'int_rate': float(int_rate),
        'installment': float(installment), 'grade': float(grade_map[grade_letter]),
        'emp_length': float(emp_length), 'home_ownership': home_ownership,
        'annual_inc': float(annual_inc), 'verification_status': verification_status,
        'purpose': purpose, 'dti': float(dti), 'open_acc': float(open_acc),
        'pub_rec': float(pub_rec), 'revol_util': float(revol_util), 'total_acc': float(total_acc)
    }

    # Convert to dataframe and generate the dummy variables
    df_in = pd.DataFrame([input_dict])
    df_in = pd.get_dummies(df_in, columns=['home_ownership', 'verification_status', 'purpose'])
    
    # MAGIC FIX: Force this new dataframe to exactly match the scaler's trained columns
    expected_cols = scaler.feature_names_in_
    df_aligned = df_in.reindex(columns=expected_cols, fill_value=0)
    
    # Scale the aligned data
    scaled_applicant = scaler.transform(df_aligned)[0]
    
    # Extract ALL 4,000 learned weights to calculate full uncertainty
    all_weights = trace.posterior['weights'].values.reshape(-1, len(expected_cols))
    all_biases = trace.posterior['bias'].values.flatten()
    
    risk_scores = np.dot(all_weights, scaled_applicant) + all_biases
    probabilities = 1 / (1 + np.exp(-risk_scores))
    final_risk = np.mean(probabilities) * 100
    
    # --- 5. Display Dashboard Results ---
    st.subheader("📊 Assessment Results")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric("Predicted Default Probability", f"{final_risk:.2f}%")
        
        if final_risk < 5.0:
            st.success("Decision: Approved - 8% Rate (Low Risk)")
        elif final_risk < 15.0:
            st.warning("Decision: Approved - 14% Rate (Medium Risk)")
        elif final_risk < 30.0:
            st.error("Decision: Approved - 22% Rate (High Risk)")
        else:
            st.error("Decision: Denied - Risk too high")
            
        st.progress(min(final_risk / 100.0, 1.0))
        
    with res_col2:
        st.write("##### Bayesian Uncertainty Distribution")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(probabilities * 100, bins=40, color='#4A90E2', edgecolor='white', alpha=0.8)
        ax.axvline(final_risk, color='red', linestyle='dashed', linewidth=2, label=f'Mean Risk: {final_risk:.1f}%')
        ax.set_xlabel("Probability of Default (%)")
        ax.set_ylabel("Frequency (Simulations)")
        ax.legend()
        
        # Make the chart background transparent to match Streamlit
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white')
        st.pyplot(fig)
        
    st.divider()
    
    # Add the MCMC Trace diagnostics at the bottom
    st.subheader("🧠 Engine Diagnostics (Bias Trace)")
    fig_trace = az.plot_trace(trace, var_names=["bias"])[0,0].figure
    st.pyplot(fig_trace)