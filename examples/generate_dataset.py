from sklearn.datasets import make_moons
import pandas as pd

# Genera il dataset
X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

# Crea un DataFrame
df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
df['class'] = y

# Salva il DataFrame in un file CSV
df.to_csv('moons.csv', index=False)

print("Dataset saved to 'moons.csv'")
