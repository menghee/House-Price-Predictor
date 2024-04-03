import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

# Load the salmon dataset
df = pd.read_json('Data/Salmon_Price.json')
# Increase the price range in the training dataset
#df['Price(in $)'] *= 100

# Extract features (X) and target variable (y)
X = df[['Length(in cm)', 'Species', 'Type']]
y = df['Price(in $)'] 


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Map 'Type' to numerical representation
type_mapping = {
    'Wild': 0,
    'Farm': 1
}
X_train['Type'] = X_train['Type'].map(type_mapping)

# One-hot encode the 'Species' feature
# Fit and transform with training data only
species_encoder = OneHotEncoder()
species_encoded_train = species_encoder.fit_transform(X_train[['Species']]).toarray() 
species_cols = species_encoder.get_feature_names_out(['Species'])  # Get encoded column names

# Create DataFrame with encoded features
X_train_encoded = pd.DataFrame(
    np.concatenate([
        X_train['Length(in cm)'].values.reshape(-1, 1), 
        species_encoded_train, 
        X_train['Type'].values.reshape(-1, 1)
    ], axis=1),
    columns=['Length(in cm)'] + list(species_cols) + ['Type']
)

print("Number of features (X_train_encoded shape):", X_train_encoded.shape) 
print("Column names:", X_train_encoded.columns)

# Create and train the linear regression model
#model = LinearRegression()
model = DecisionTreeRegressor()
model.fit(X_train_encoded, y_train)

# Evaluate model performance (optional)
# Transform test data and evaluate
species_encoded_test = species_encoder.transform(X_test[['Species']]).toarray()
X_test_encoded = pd.DataFrame(
    np.concatenate([
        X_test['Length(in cm)'].values.reshape(-1, 1), 
        species_encoded_test, 
        X_test['Type'].map(type_mapping).values.reshape(-1, 1)
    ], axis=1),
    columns=['Length(in cm)'] + list(species_cols) + ['Type']
)
mse = np.mean((model.predict(X_test_encoded) - y_test)**2)  
print('MSE:', mse) 

# Save the trained model and the encoder
import pickle 
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(species_encoder, open('species_encoder.pkl', 'wb')) 
