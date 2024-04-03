from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder 

app = Flask(__name__)

# Load the trained model and encoder
model = pickle.load(open('model.pkl', 'rb'))
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
    species = request.form['species']
    salmon_type = request.form['type']

    new_data = {'Length(in cm)': [length], 'Species': [species], 'Type': [salmon_type]}
    df = pd.DataFrame(new_data)

    df['Type'] = df['Type'].map(type_mapping)
    species_encoded = species_encoder.transform(df[['Species']]).toarray()

    encoded_features =  df['Length(in cm)'].to_list() + species_encoded.tolist()[0] + [df['Type'].iloc[0]] 

    # Get column names from model training (you may need to adjust these)
    column_names = ['Length(in cm)', 'Species_Chinook', 'Species_Chum', 'Species_Coho', 'Species_Pink', 'Species_Sockeye', 'Type'] 
    print("Length of encoded_features:", len(encoded_features)) 
    print("column_names:", column_names)

    input_df = pd.DataFrame([encoded_features], columns=column_names)
    prediction = model.predict(input_df)[0]

    return render_template('index.html', prediction_text='Predicted Price of Salmon: $ {:.2f}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)
