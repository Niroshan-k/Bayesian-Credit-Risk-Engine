from src.load_data import fetch_data
from src.cleaning import clean_data
from src.model import train
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

def assign_interest_rate(risk_percentage):
    if risk_percentage < 5.0:
        return f"Approved: 8% Rate (Low Risk: {risk_percentage:.1f}%)"
    elif risk_percentage < 15.0:
        return f"Approved: 14% Rate (Medium Risk: {risk_percentage:.1f}%)"
    elif risk_percentage < 30.0:
        return f"Approved: 22% Rate (High Risk: {risk_percentage:.1f}%)"
    else:
        return f"Denied: Risk too high ({risk_percentage:.1f}%)"

def main():
    print("loan csv loading...")
    loan_data = fetch_data()
    print("data loaded successfully!")
    
    print("cleaning data...")
    cleaned_data = clean_data(loan_data)
    print("data cleaned successfully!")

    print(cleaned_data.head(5))
    #print(cleaned_data.info())

    # 6. Visualize the Uncertainty
    trace = train(cleaned_data)
    az.plot_trace(trace)
    plt.tight_layout()
    plt.show()

    #test
    X = cleaned_data.drop('default', axis=1).values
    new_applicant = X[0]

    # 2. Extract the thousands of learned weights from your trace
    learned_weights = trace.posterior['weights'].values.reshape(-1, X.shape[1])
    learned_bias = trace.posterior['bias'].values.flatten()

    # 3. Calculate their risk across all 4,000 simulations
    risk_scores = np.dot(new_applicant, learned_weights.T) + learned_bias
    probabilities = 1 / (1 + np.exp(-risk_scores)) # The Sigmoid Squeeze

    # 4. Get the final average risk percentage
    final_risk = np.mean(probabilities) * 100
    print(f"Predicted Default Risk: {final_risk:.2f}%")

    # Test applicant's risk
    decision = assign_interest_rate(final_risk)
    print(decision)

if __name__ == "__main__":
    main()