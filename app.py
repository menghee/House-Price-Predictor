from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder 
from flask import jsonify # Import jsonify

app = Flask(__name__)

# Load the trained model and encoder
multi_variant_model = pickle.load(open('model.pkl', 'rb'))
simple_model = pickle.load(open('model3.pkl', 'rb'))
species_encoder = pickle.load(open('species_encoder.pkl', 'rb'))

# Type Mapping (Assuming numerical encoding from 'model.py')
type_mapping = {
    'Wild': 0,
    'Farm': 1
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    length = float(request.form['length'])
    model_type = request.form.get('modelType', 'multi')  # Get the model type (default to multi)

    if model_type == 'simple':
        # Predict using the simple model
        prediction = simple_model.predict([[length]])[0]
    else:
        # Predict using the multi-variant model
        species = request.form['species']
        salmon_type = request.form['type']
        
        new_data = {'Length(in cm)': [length], 'Species': [species], 'Type': [salmon_type]}
        df = pd.DataFrame(new_data)

        df['Type'] = df['Type'].map(type_mapping)
        species_encoded = species_encoder.transform(df[['Species']]).toarray()

        encoded_features =  df['Length(in cm)'].to_list() + species_encoded.tolist()[0] + [df['Type'].iloc[0]] 

        # Get column names from model training (you may need to adjust these)
        column_names = ['Length(in cm)', 'Species_Chinook', 'Species_Chum', 'Species_Coho', 'Species_Pink', 'Species_Sockeye', 'Type'] 
        input_df = pd.DataFrame([encoded_features], columns=column_names)
        prediction = multi_variant_model.predict(input_df)[0]

    return jsonify({'prediction_text': '$ {:.2f}'.format(prediction)})  # Return only the predicted price

if __name__ == '__main__':
    app.run(debug=True)