import pandas as pd
from sys import exit

def candidate_elimination(data):
    specific_hypothesis = ['0'] * (data.shape[1] - 1)  
    general_hypothesis = [['?'] * (data.shape[1] - 1)] 
    for index, row in data.iterrows():
        if row['Play'] == 'Yes': 
            for i in range(len(specific_hypothesis)):
                if specific_hypothesis[i] == '0':  
                    specific_hypothesis[i] = row.iloc[i]
                elif specific_hypothesis[i] != row.iloc[i]:  
                    specific_hypothesis[i] = '?'
            general_hypothesis = [g for g in general_hypothesis if all(
                g[i] == '?' or g[i] == row.iloc[i] or specific_hypothesis[i] == '?'
                for i in range(len(g))
            )]

        else: 
            new_general_hypotheses = []
            for i in range(len(specific_hypothesis)):
                if specific_hypothesis[i] == '?':
                    continue 
                elif specific_hypothesis[i] != row.iloc[i]:
                    new_hypothesis = specific_hypothesis.copy()
                    new_hypothesis[i] = '?'
                    new_general_hypotheses.append(new_hypothesis)

            general_hypothesis.extend([h for h in new_general_hypotheses if all(
                h[j] == '?' or h[j] == row.iloc[j] or specific_hypothesis[j] == '?'
                for j in range(len(h))
            )])
   
    general_hypothesis = [g for g in general_hypothesis if not any(
        all(g[i] == '?' or g[i] == other[i] for i in range(len(g))) and g != other
        for other in general_hypothesis
    )]

    return specific_hypothesis, general_hypothesis

file_path = r'./weather_forecast.csv'  
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()

required_columns = ['Outlook', 'Temperature', 'Humidity', 'Wind', 'Play']
if not all(column in data.columns for column in required_columns):
    print("Error: The dataset must include the following columns:")
    print(required_columns)
    exit()

specific_hypothesis, general_hypothesis = candidate_elimination(data)

print("Specific Hypothesis:", specific_hypothesis)
print("General Hypotheses:", general_hypothesis)
