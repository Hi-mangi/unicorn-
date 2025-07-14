# unicorn-
unicorn startup valuation model
# Import libraries
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Generate synthetic data
random.seed(42)
np.random.seed(42)

n_samples = 300

industries = ['Fintech', 'E-commerce', 'AI', 'HealthTech', 'EdTech']
countries = ['USA', 'India', 'UK', 'China', 'Germany']

data = {
    'Industry': np.random.choice(industries, n_samples),
    'Country': np.random.choice(countries, n_samples),
    'Year_Joined': np.random.randint(2010, 2024, n_samples),
    'Num_Investors': np.random.randint(1, 50, n_samples),
}

# Generate synthetic valuations (in billions) with a pattern
base_vals = []
for i in range(n_samples):
    base = 0.5
    base += industries.index(data['Industry'][i]) * 2
    base += countries.index(data['Country'][i]) * 1.5
    base += (data['Year_Joined'][i] - 2010) * 0.3
    base += data['Num_Investors'][i] * 0.1
    noise = np.random.normal(0, 2)
    base_vals.append(round(base + noise, 2))

data['Valuation'] = base_vals

df = pd.DataFrame(data)

# 2. Split features and target
X = df.drop('Valuation', axis=1)
y = df['Valuation']

# 3. Preprocessing pipeline
numeric_features = ['Year_Joined', 'Num_Investors']
categorical_features = ['Industry', 'Country']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# 4. Define model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model
model.fit(X_train, y_train)

# 7. Evaluate model
y_pred = model.predict(X_test)

print("R2 Score:", round(r2_score(y_test, y_pred), 3))
print("MSE:", round(mean_squared_error(y_test, y_pred), 3))

# 8. Plot actual vs predicted
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Valuation (Billion $)")
plt.ylabel("Predicted Valuation (Billion $)")
plt.title("Actual vs Predicted Unicorn Startup Valuations")
plt.grid(True)
plt.show()
