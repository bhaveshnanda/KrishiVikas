import pandas as pd
import random

# Define possible fertilizers and their typical ranges for nitrogen, potassium, and phosphorous
fertilizers = [
    ("Urea", (35, 45), (0, 0), (0, 0)),
    ("DAP", (10, 15), (35, 45), (35, 45)),
    ("Fourteen-Thirty Five-Fourteen", (5, 10), (28, 35), (28, 35)),
    ("Twenty Eight-Twenty Eight", (20, 25), (0, 0), (15, 25)),
    ("Seventeen-Seventeen-Seventeen", (10, 20), (10, 20), (10, 20)),
    ("Twenty-Twenty", (5, 15), (5, 15), (5, 15)),
    ("Ten-Twenty Six-Twenty Six", (5, 15), (15, 25), (15, 25))
]

data = []

# Generate 2000 random rows of fertilizer data
for _ in range(2000):
    fertilizer = random.choice(fertilizers)
    nitrogen = random.randint(*fertilizer[1])
    potassium = random.randint(*fertilizer[2])
    phosphorous = random.randint(*fertilizer[3])
    
    data.append([nitrogen, potassium, phosphorous, fertilizer[0]])

# Convert to a DataFrame
df = pd.DataFrame(data, columns=["Nitrogen", "Potassium", "Phosphorous", "Fertilizer Name"])

# Save as CSV
df.to_csv("fertilizer_data.csv", index=False)

print("Generated 2000 rows of data and saved to 'fertilizer_data.csv'.")
