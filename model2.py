import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load the salmon dataset
df = pd.read_json('Data/Salmon_Price.json')

# Extract features (X) and target variable (y)
X = df[['Length(in cm)', 'Species', 'Type']]
y = df['Price(in $)'] 

# Map 'Type' to numerical representation
type_mapping = {
    'Wild': 0,
    'Farm': 1
}
X['Type'] = X['Type'].map(type_mapping)

# One-hot encode the 'Species' feature
# ... Make sure your dataset has 'Species_Chinook' instances before fitting 
species_encoder = OneHotEncoder()
species_encoded = species_encoder.fit_transform(df[['Species']]).toarray() # Fit with all species
species_cols = species_encoder.get_feature_names_out(['Species'])  # Get encoded column names

X = pd.DataFrame(np.concatenate([X['Length(in cm)'].values.reshape(-1, 1), species_encoded, X['Type'].values.reshape(-1, 1)], axis=1),
                 columns=['Length(in cm)'] + list(species_cols) + ['Type'])

print("Number of features (X shape):", X.shape) 
print("Column names:", X.columns)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model performance (optional)
mse = np.mean((model.predict(X_test) - y_test)**2)  
print('MSE:', mse) 

# Save the trained model and the encoder
import pickle 
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(species_encoder, open('species_encoder.pkl', 'wb')) 
