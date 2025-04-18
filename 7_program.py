import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

heart_disease = pd.read_csv('heart.csv')

print('Sample instances from the dataset:')
print(heart_disease.head())

print('\nAttributes and data types:')
print(heart_disease.dtypes)

model = BayesianModel([
    ('age', 'output'),
    ('sex', 'output'),
    ('cp', 'output'),
    ('trtbps', 'output'),
    ('chol', 'output'),
    ('restecg', 'output')
])

heart_disease_sampled = heart_disease.sample(frac=0.1, random_state=42)

print('\nLearning CPDs using Maximum Likelihood Estimators...')
model.fit(heart_disease_sampled, estimator=MaximumLikelihoodEstimator)

print('\nInferencing with Bayesian Network:')
heart_disease_infer = VariableElimination(model)

print('\n1. Probability of Heart Disease given evidence (restecg = 1):')
query1 = heart_disease_infer.query(variables=['output'], evidence={'restecg': 1})
print(query1)

print('\n2. Probability of Heart Disease given evidence (cp = 2):')
query2 = heart_disease_infer.query(variables=['output'], evidence={'cp': 2})
print(query2)
