import pandas as pd
import pickle

# Load model
with open('model_checkpoint.pkl', 'rb') as f:
    clf = pickle.load(f)

# Load data
csv_path = '../../final.csv'
df = pd.read_csv(csv_path)
X = df[['id']]

# Predict
preds = clf.predict(X)

# Save predictions
out_df = pd.DataFrame({'id': df['id'], 'label': preds})
out_df.to_csv('predictions.csv', index=False)
print('Predictions saved to predictions.csv') 