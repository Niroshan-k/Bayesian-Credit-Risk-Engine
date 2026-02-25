import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def clean_data(data):
    
    df = data.copy()

    keep_columns = [
        'loan_amnt',          # How much they asked for
        'term',               # 36 or 60 months
        'int_rate',           # Interest rate assigned
        'installment',        # Monthly payment
        'grade',              # LendingClub's internal risk grade
        'emp_length',         # Years employed
        'home_ownership',     # Rent, Own, Mortgage
        'annual_inc',         # Annual income
        'verification_status',# Is income verified?
        'purpose',            # What the loan is for
        'dti',                # Debt-to-Income ratio (Crucial!)
        'open_acc',           # Number of open credit lines
        'pub_rec',            # Derogatory public records (bankruptcies)
        'revol_util',         # Credit card utilization rate
        'total_acc',          # Total credit lines ever opened
        'loan_status'         # Our target variable to predict!
    ]

    df = df[keep_columns]

    # 1. Keep only loans that have a final outcome
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]

    # 2. Create the target 'default' column (1 = Default, 0 = Paid)
    df['default'] = (df['loan_status'] == 'Charged Off').astype(int)

    # 3. Drop the loan status text column
    df = df.drop('loan_status', axis=1)

    df['term'] = df['term'].astype(str).str.extract('(\d+)').astype(float)
    df['emp_length'] = df['emp_length'].astype(str).str.extract('(\d+)').fillna(0).astype(float)
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade'] = df['grade'].map(grade_map)
    df = pd.get_dummies(df, columns=['home_ownership', 'verification_status', 'purpose'], drop_first=True)
    
    # Ensure everything is strictly numeric
    df = df.astype(float)
    df = df.dropna()

    # Take a small random sample to test the model
    df = df.sample(n=2000, random_state=42)

    # Scale all columns EXCEPT our target 'default'
    cols_to_scale = df.columns.drop('default')
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    # Save the translator for the dashboard
    os.makedirs("model", exist_ok=True)
    joblib.dump(scaler, "model/scaler.pkl")

    return df