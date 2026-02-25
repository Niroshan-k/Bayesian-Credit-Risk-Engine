import os
import pymc as pm
import numpy as np
import arviz as az

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "sme_risk_model.nc")

def train(df):
    # Separate your features (X) from your target (y)
    X = df.drop('default', axis=1).values
    y = df['default'].values

    with pm.Model() as model:
        weights = pm.Normal('weights', mu=0, sigma=10, shape=X.shape[1])
        bias = pm.Normal('bias', mu=0, sigma=10)

        risk_score = pm.math.dot(X, weights) + bias
        # sigmoid function to convert risk score to probability
        probability = pm.math.invlogit(risk_score)

        # Bernoulli likelihood for the observed data : like error function
        likelihood = pm.Bernoulli('likelihood', p=probability, observed=y)

        if os.path.exists(MODEL_PATH):
            print("Loading saved model...")
            trace = az.from_netcdf(MODEL_PATH)
        else:
            print("Starting the simulations... training the model")
            trace = pm.sample(draws=1000, tune=1000, cores=1,return_inferencedata=True)
            
            os.makedirs(MODEL_DIR, exist_ok=True)
            az.to_netcdf(trace, MODEL_PATH)
            print(f"Model saved successfully to {MODEL_PATH}!")

        return trace