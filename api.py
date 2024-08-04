import numpy as np
from flask import Flask, request, jsonify, render_template_string
import pickle
import pandas as pd
from car_data_prep import prepare_data 
import os

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/')
def index():
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Car Price Prediction</title>
        <link href="https://fonts.googleapis.com/css2?family=Heebo:wght@300&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Heebo', sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                background-color: #f0f0f0;
                margin: 0;
                padding: 0;
            }
            .container {
                background-color: #fff;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                padding: 20px;
                margin-top: 50px;
                max-width: 500px;
                width: 100%;
                position: relative;
            }
            .background-layer {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: url('https://yahad-motors.co.il/wp-content/uploads/2022/12/shutterstock_797260003-min-1536x932.jpg') no-repeat center center/cover;
                opacity: 0.2; /* Change the opacity as needed */
                border-radius: 10px;
                z-index: 0;
            }
            .content {
                position: relative;
                z-index: 1;
            }
            h1 {
                color: #007bff;
            }
            h2 {
                color: #555;
            }
            label {
                display: block;
                margin-top: 10px;
                font-weight: bold;
            }
            .required:after {
                content: '*';
                color: red;
                margin-left: 5px;
            }
            input[type="text"], select, input[list] {
                width: calc(100% - 22px);
                padding: 10px;
                margin-top: 5px;
                margin-bottom: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            button {
                background-color: #007bff;
                color: #fff;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                transition: background-color 0.3s ease;
            }
            button:hover {
                background-color: #0056b3;
            }
            .error-message {
                color: red;
                display: none;
                margin-top: 10px;
            }
            #prediction-text {
                margin-top: 20px;
                font-size: 18px;
                font-weight: bold;
                color: #28a745;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="background-layer"></div>
            <div class="content">
                <h1>Car Price Prediction</h1>
                <h2>Enter the car's data: </h2>
                <form id="measurement-form">
                    <label for="manufactor" class="required">Manufactor:</label>
                    <input list="manufactors" id="manufactor" name="manufactor">
                    <datalist id="manufactors">
                        <option value="אאודי">
                        <option value="אברת'">
                        <option value="אוטוביאנקי">
                        <option value="אולדסמוביל">
                        <option value="אוסטין">
                        <option value="אופל">
                        <option value="אינפיניטי">
                        <option value="אלפא רומיאו">
                        <option value="אם. ג'י. / MG">
                        <option value="ב.מ.וו">
                        <option value="ביואיק">
                        <option value="גרייט וול / G.O">
                        <option value="דאצ'יה">
                        <option value="דודג'">
                        <option value="דייהו">
                        <option value="דייהטסו">
                        <option value="הונדה">
                        <option value="וולוו">
                        <option value="טויוטה">
                        <option value="טסלה">
                        <option value="יגואר">
                        <option value="יונדאי">
                        <option value="לאדה">
                        <option value="לינקולן">
                        <option value="לנצ'יה">
                        <option value="לקסוס">
                        <option value="מאזדה">
                        <option value="מזראטי">
                        <option value="מיני">
                        <option value="מיצובישי">
                        <option value="מרצדס">
                        <option value="ניסאן">
                        <option value="סאאב">
                        <option value="סאנגיונג">
                        <option value="סובארו">
                        <option value="סוזוקי">
                        <option value="סיאט">
                        <option value="סיטרואן">
                        <option value="סמארט">
                        <option value="סקודה">
                        <option value="פולקסווגן">
                        <option value="פונטיאק">
                        <option value="פורד">
                        <option value="פורשה">
                        <option value="פיאט">
                        <option value="פיג'ו">
                        <option value="פרארי">
                        <option value="קאדילק">
                        <option value="קיה">
                        <option value="קרייזלר">
                        <option value="רובר">
                        <option value="רנו">
                        <option value="שברולט">
                    </datalist>
                    <label for="year" class="required">Year:</label>
                    <input type="text" id="year" name="Year">
                    <label for="model" class="required">Model:</label>
                    <input type="text" id="model" name="model">
                    <label for="hand" class="required">Hand:</label>
                    <input type="text" id="hand" name="Hand">
                    <label for="gear" class="required">Gear:</label>
                    <select id="gear" name="Gear">
                        <option value="">Select Gear Type</option>
                        <option value="אוטומטית">אוטומטית</option>
                        <option value="ידנית">ידנית</option>
                        <option value="טיפטרוניק">טיפטרוניק</option>
                        <option value="רובוטית">רובוטית</option>
                    </select>
                    <label for="capacity_Engine" class="required">Capacity Engine:</label>
                    <input type="text" id="capacity_Engine" name="capacity_Engine">
                    <label for="engine_type" class="required">Engine Type:</label>
                    <select id="engine_type" name="Engine_type">
                        <option value="">Select Engine Type</option>
                        <option value="בנזין">בנזין</option>
                        <option value="דיזל">דיזל</option>
                        <option value="גז">גז</option>
                        <option value="היברידי">היברידי</option>
                        <option value="חשמלי">חשמלי</option>
                    </select>
                    <label for="prev_ownership" class="required">Previous Ownership:</label>
                    <select id="prev_ownership" name="Prev_ownership">
                        <option value="">Select Previous Ownership</option>
                        <option value="פרטית">פרטית</option>
                        <option value="חברה">חברה</option>
                        <option value="השכרה">השכרה</option>
                        <option value="ליסינג">ליסינג</option>
                        <option value="מונית">מונית</option>
                        <option value="לימוד נהיגה">לימוד נהיגה</option>
                        <option value="ייבוא אישי">ייבוא אישי</option>
                        <option value="ממשלתי">ממשלתי</option>
                        <option value="אחר">אחר</option>
                    </select>
                    <label for="curr_ownership" class="required">Current Ownership:</label>
                    <select id="curr_ownership" name="Curr_ownership">
                        <option value="">Select Current Ownership</option>
                        <option value="פרטית">פרטית</option>
                        <option value="חברה">חברה</option>
                        <option value="השכרה">השכרה</option>
                        <option value="ליסינג">ליסינג</option>
                        <option value="מונית">מונית</option>
                        <option value="לימוד נהיגה">לימוד נהיגה</option>
                        <option value="ייבוא אישי">ייבוא אישי</option>
                        <option value="ממשלתי">ממשלתי</option>
                        <option value="אחר">אחר</option>
                    </select>
                    <label for="description" class="required">Description:</label>
                    <input type="text" id="description" name="Description">
                    <label for="km" class="required">Kilometers:</label>
                    <input type="text" id="km" name="Km">
                    <label for="test" class="required">Test (DD/MM/YYYY):</label>
                    <input type="text" id="test" name="Test" placeholder="DD/MM/YYYY">
                    <button type="submit">Calculate Price</button>  
                    <p class="error-message" id="error-message">חסר פרטים, אנא מלא את כל השדות המסומנים</p>
                </form>
                <h3 id="prediction-text"></h3>
            </div>
        </div>
        <script>
            document.getElementById('measurement-form').addEventListener('submit', function(event) {
                event.preventDefault();
                if (validateForm()) {
                    const formData = new FormData(this);
                    fetch('/predict', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('prediction-text').textContent = data.prediction;
                    })
                    .catch(error => {
                        console.error('Error:', error);
                    });
                }
            });

            function validateForm() {
                const requiredFields = ['manufactor', 'year', 'model', 'hand', 'gear', 'capacity_Engine', 'engine_type', 'prev_ownership', 'curr_ownership', 'description', 'km', 'test'];
                let valid = true;
                requiredFields.forEach(function(field) {
                    const input = document.getElementById(field);
                    if (!input.value.trim()) {
                        valid = false;
                        input.previousElementSibling.style.color = 'red';
                    } else {
                        input.previousElementSibling.style.color = 'black';
                    }
                });
                const testField = document.getElementById('test');
                const testPattern = /^\d{2}\/\d{2}\/\d{4}$/;
                if (!testPattern.test(testField.value.trim())) {
                    valid = false;
                    testField.previousElementSibling.style.color = 'red';
                } else {
                    testField.previousElementSibling.style.color = 'black';
                }
                if (!valid) {
                    document.getElementById('error-message').style.display = 'block';
                } else {
                    document.getElementById('error-message').style.display = 'none';
                }
                return valid;
            }
        </script>
    </body>
    </html>
    '''
    return render_template_string(html_content)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'manufactor': request.form.get('manufactor', ''),
            'Year': request.form.get('Year', ''),
            'model': request.form.get('model', ''),
            'Hand': request.form.get('Hand', ''),
            'Gear': request.form.get('Gear', ''),
            'capacity_Engine': request.form.get('capacity_Engine', ''),
            'Engine_type': request.form.get('Engine_type', ''),
            'Prev_ownership': request.form.get('Prev_ownership', ''),
            'Curr_ownership': request.form.get('Curr_ownership', ''),
            'Description': request.form.get('Description', ''),
            'Km': request.form.get('Km', ''),
            'Test': request.form.get('Test', '')
        }

        # List of all required columns
        columns = ['manufactor', 'Year', 'model', 'Hand', 'Gear', 'capacity_Engine', 'Engine_type', 
                   'Prev_ownership', 'Curr_ownership', 'Area', 'City', 'Price', 'Pic_num', 'Cre_date', 
                   'Repub_date', 'Description', 'Color', 'Km', 'Test', 'Supply_score']

        # Create DataFrame with the required columns and leave columns not in the form as empty
        input_df = pd.DataFrame(columns=columns)
        input_df.loc[0] = np.nan  # Add empty row

        # Insert data from the form into the DataFrame
        for key, value in input_data.items():
            input_df.at[0, key] = value

        # Convert to numeric values for appropriate columns
        numeric_columns = ['Year', 'Hand', 'capacity_Engine', 'Km']
        for col in numeric_columns:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # Prepare data using prepare_data function
        prepared_data = prepare_data(input_df)

        # Get the ColumnTransformer from the trained model
        preprocessor = model.named_steps['preprocessor']

        # Process the data using the ColumnTransformer from the trained model
        processed_data = preprocessor.transform(prepared_data)

        # Predict the price
        prediction = model.named_steps['model'].predict(processed_data)[0]

        output = round(prediction, 2)

        return jsonify(prediction=f'Predicted Price: {output}')
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify(prediction=f"Error during prediction: {e}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
