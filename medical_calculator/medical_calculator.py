import xgboost as xgb # import first to avoid threding issue ('1 leaked semaphore objects') 
import numpy as np
from flask import Flask, request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append('/Users/mirandazheng/Desktop/first/ensure')
sys.path.append('/Users/mirandazheng/Desktop/first')
import dummy_coding
import imp_norm
import ensure.medical_calculator.cox_model as cox_model
import ensure.medical_calculator.deepsurv_model as deepsurv_model
from torchtuple_cox import _CoxPHBase
from pycox.models import loss as pycox_loss
from torchtuple_pmf import PMFBase
import ensure.medical_calculator.deephit_model as deephit_model
from pycox import models
import ensure.medical_calculator.xgboost_model as xgboost_model




################## note ########################################################################
'''cpu only'''


################## load the model ########################################################################

# has to be in global place, otherwise leads problem with pickle loading
class CoxPH_new(_CoxPHBase):
        def __init__(self, net, optimizer=None, device=None, loss=None, scheduler=None):
            if loss is None:
                loss = pycox_loss.CoxPHLoss()
            super().__init__(net, loss, optimizer, device)
            self.scheduler = scheduler


class DeepHitSingle_new(PMFBase):

    def __init__(self, net, optimizer=None, device=None, duration_index=None, alpha=0.2, sigma=0.1, loss=None, scheduler=None):
        if loss is None:
            loss = pycox_loss.DeepHitSingleLoss(alpha, sigma)
        super().__init__(net, loss, optimizer, device, duration_index)
        self.scheduler = scheduler

    def make_dataloader(self, data, batch_size, shuffle, num_workers=0):
        dataloader = super().make_dataloader(data, batch_size, shuffle, num_workers,
                                             make_dataset=models.data.DeepHitDataset)
        return dataloader
    
    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        dataloader = super().make_dataloader(input, batch_size, shuffle, num_workers)
        return dataloader
    

################## make the website ########################################################################


# Initialize Flask app
app = Flask(__name__)


# Define the home route with an HTML form for input
@app.route('/')
def home():
    return '''
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Recurrence & Survival Calculator for Patients after Curative-intent Esophagectomy</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f9f9f9;
            }
            h1 {
                text-align: center;
                font-size: 28px;
            }
            form {
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
            }
            table {
                border-spacing: 15px;
                width: 100%;
                max-width: 1200px;
                margin: 0 auto;
            }
            td {
                padding: 5px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            select, input {
                width: 100%;
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            button {
                margin-top: 20px;
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
            /* Add custom colors */
            #Clinical\ Endpoint {
                background-color: #0D4A70; /* yellow color */
                color: white;
                border: 1px solid #0D4A70;
                border-radius: 5px;
            }
            #Survival\ Model {
                background-color: #0D4A70; /* yellow color */
                color: white;
                border: 1px solid #0D4A70;
                border-radius: 5px;
            }
            #Clinical\ Endpoint:hover, #Survival\ Model:hover {
                background-color: #0D4A70; /* color shade on hover */
            }
        </style>
    </head>
    <body>
        <h1>Recurrence & Survival Calculator for Patients after Curative-intent Esophagectomy</h1>
        <form action="/predict" method="post">
            <table>
                <tr>
                    <td>
                        <label for="Sex">Sex</label>
                        <select id="Sex" name="Sex" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">Female</option>
                            <option value="1">Male</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                    <td>
                        <label for="Age at diagnosis">Age at Diagnosis</label>
                        <input type="text" id="Age at diagnosis" name="Age at diagnosis" placeholder="Enter age" required>
                    </td>
                    <td>
                        <label for="ASA grade">ASA Grade</label>
                        <select id="ASA grade" name="ASA grade" required>
                            <option value="" disabled selected>Select</option>
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                    <td>
                        <label for="Barrett's esophagus">Barrett's Esophagus</label>
                        <select id="Barrett's esophagus" name="Barrett's esophagus" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">No</option>
                            <option value="1">Yes</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                    <td>
                        <label for="Histologic type">Histologic Type</label>
                        <select id="Histologic type" name="Histologic type" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">Adenocarcinoma</option>
                            <option value="1">Squamous Cell Carcinoma</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                </tr>
                <tr>
                    <td>
                        <label for="Clinical T stage">Clinical T Stage</label>
                        <select id="Clinical T stage" name="Clinical T stage" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                            <option value="4">4</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                    <td>
                        <label for="Clinical N stage">Clinical N stage</label>
                        <select id="Clinical N stage" name="Clinical N stage" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                    <td>
                        <label for="Clinical M stage">Clinical M stage</label>
                        <select id="Clinical M stage" name="Clinical M stage" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                    <td>
                        <label for="Clinical Differentiation">Clinical Differentiation</label>
                        <select id="Clinical Differentiation" name="Clinical Differentiation" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                            <option value="4">4</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                    <td>
                        <label for="Tumour site">Tumour site</label>
                        <select id="Tumour site" name="Tumour site" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                </tr>
                <tr>
                    <td>
                        <label for="Distal margin positive">Distal margin positive</label>
                        <select id="Distal margin positive" name="Distal margin positive" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                    <td>
                         <label for="Proximal margin positive">Proximal margin positive</label>
                        <select id="Proximal margin positive" name="Proximal margin positive" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                    <td>
                        <label for="Radial margin positive">Radial margin positive</label>
                        <select id="Radial margin positive" name="Radial margin positive" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="NA">NA</option>
                        </select>

                    </td>
                    <td>
                        <label for="Pathologic T stage">Pathologic T stage</label>
                        <select id="Pathologic T stage" name="Pathologic T stage" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                            <option value="4">4</option>
                            <option value="5">5</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                    <td>
                        <label for="Pathologic N stage">Pathologic N stage</label>
                        <select id="Pathologic N stage" name="Pathologic N stage" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                
                </tr>
                <tr>
                     <td>
                        <label for="Pathologic M Stage">Pathologic M Stage</label>
                        <select id="Pathologic M Stage" name="Pathologic M Stage" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
        
                    <td>
                        <label for="Differentiation">Differentiation</label>
                        <select id="Differentiation" name="Differentiation" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                            <option value="4">4</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                    <td>
                        <label for="Lymphatic invasion">Lymphatic invasion</label>
                        <select id="Lymphatic invasion" name="Lymphatic invasion" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                    <td>
                        <label for="Venous invasion">Venous invasion</label>
                        <select id="Venous invasion" name="Venous invasion" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                    <td>
                        <label for="Perineural invasion">Perineural invasion</label>
                        <select id="Perineural invasion" name="Perineural invasion" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                </tr>
                <tr>
                    
                    <td>
                        <label for="Number of nodes analyzed">Number of nodes analyzed</label>
                        <input type="text" id="Number of nodes analyzed" name="Number of nodes analyzed" required>
                    </td>

                    <td>
                        <label for="Treatment protocol">Treatment protocol</label>
                        <select id="Treatment protocol" name="Treatment protocol" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                            <option value="4">4</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                    <td>
                        <label for="Operation type">Operation type</label>
                        <select id="Operation type" name="Operation type" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                            <option value="4">4</option>
                            <option value="5">5</option>
                            <option value="6">6</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                    <td>
                        <label for="Approach">Approach</label>
                        <select id="Approach" name="Approach" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                            <option value="4">4</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                    <td>
                        <label for="Robotic assisted">Robotic assisted</label>
                        <select id="Robotic assisted" name="Robotic assisted" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                </tr>
                <tr>

                    <td>
                        <label for="Clavien-Dindo Grade">Clavien-Dindo Grade</label>
                        <select id="Clavien-Dindo Grade" name="Clavien-Dindo Grade" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                            <option value="4">4</option>
                            <option value="5">5</option>
                            <option value="6">6</option>
                            <option value="7">7</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                    <td>
                        <label for="Length of stay">Length of stay</label>
                        <input type="text" id="Length of stay" name="Length of stay" required>
                    </td>
                    <td>
                        <label for="Gastroenteric leak">Gastroenteric leak</label>
                        <select id="Gastroenteric leak" name="Gastroenteric leak" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                    <td>
                        <label for="GDP USD per cap">GDP USD per cap</label>
                        <input type="text" id="GDP USD per cap" name="GDP USD per cap" required>
                    </td>
                    <td>
                        <label for="H volutary">H volutary</label>
                        <input type="text" id="H volutary" name="H volutary" required>
                    </td>
                </tr>
                <tr>
                    
                    <td>
                        <label for="HE Compulsory">HE Compulsory</label>
                        <input type="text" id="HE Compulsory" name="HE Compulsory" required>
                    </td>
                    <td>
                        <label for="Cancer cases per year">Cancer cases per year</label>
                        <select id="Cancer cases per year" name="Cancer cases per year" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0-10</option>
                            <option value="1">10-20</option>
                            <option value="2">20-30</option>
                            <option value="3">30-40</option>
                            <option value="4">40-50</option>
                            <option value="5">50-60</option>
                            <option value="6">>60</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                    <td>
                        <label for="Consultant Volume SPSS">Consultant Volume SPSS</label>
                        <select id="Consultant Volume SPSS" name="Consultant Volume SPSS" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                    <td>
                        <label for="Intensive Surveillance">Intensive Surveillance</label>
                        <select id="Intensive Surveillance" name="Intensive Surveillance" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="NA">NA</option>
                        </select>
                    </td>
                </tr>
                <tr>
                    <td>
                        <label for="Clinical Endpoint">Clinical Endpoint</label>
                        <select id="Clinical Endpoint" name="Clinical Endpoint" required>
                            <option value="" disabled selected>Select</option>
                            <option value="DFS">DFS</option>
                            <option value="OS">OS</option>
                            <option value="DSS">DSS</option>
                        </select>
                    </td>
                    <td>
                        <label for="Survival Model">Survival Model</label>
                        <select id="Survival Model" name="Survival Model" required>
                            <option value="" disabled selected>Select</option>
                            <option value="CoxPH">CoxPH</option>
                            <option value="XGBoost">XGBoost</option>
                            <option value="DeepSurv">DeepSurv</option>
                            <option value="DeepHit">DeepHit</option>
                        </select>
                    </td>
                    <td colspan="2">
                        <button type="submit">Calculate</button>
                    </td>
                </tr>
            </table>
        </form>
    </body>
    </html>
    '''

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debugging: Print raw form data
        print("Form data received:", request.form)
        clinical_endpoint = request.form['Clinical Endpoint']
        survival_model = request.form['Survival Model']

        # Extract all input data from the form and store it in a dictionary
        data = {key: request.form[key] for key in request.form.keys()}
        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame([data])
        df = df.iloc[:, :-2] # exclude the last two arguments which are clinical endpoints and models
        # Replace 'NA' with np.nan for handling missing values
        df.replace('NA', np.nan, inplace=True)
        # Convert numeric columns to float leaving np.nan as is
        df = df.map(lambda x: float(x) if pd.notnull(x) else x)


        # dummy code the data
        df_dc = dummy_coding.dummy_code_data(df) 

        # impute and normalise the data
        df_imp_norm = imp_norm.imp_norm(df_dc, clinical_endpoint)


        if survival_model == 'CoxPH':

            encoded_image = cox_model.cox_model(df_imp_norm, clinical_endpoint)

        if survival_model == 'DeepSurv':

            encoded_image = deepsurv_model.deepsurv_model(df_imp_norm, clinical_endpoint)

        if survival_model == 'DeepHit':

            encoded_image = deephit_model.deephit_model(df_imp_norm, clinical_endpoint)
            
        if survival_model == 'XGBoost':

            encoded_image = xgboost_model.xgboost_model(df, clinical_endpoint)
            
        

        # Return the response
        return f'''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Prediction Result</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    flex-direction: column;
                    align-items: center; /* Center horizontally */
                    justify-content: center; /* Center vertically */
                    min-height: 100vh; /* Full height of the viewport */
                    background-color: #f9f9f9; /* Optional background color */
                }}
                h1, h2 {{
                    text-align: center;
                }}
                img {{
                    display: block;
                    margin: 20px auto; /* Center horizontally */
                    max-width: 80%; /* Adjust image size as needed */
                    height: auto;
                }}
                a {{
                    text-decoration: none;
                    color: white;
                    background-color: #4CAF50;
                    padding: 10px 20px;
                    border-radius: 5px;
                    display: inline-block;
                    margin-top: 20px;
                }}
                a:hover {{
                    background-color: #45a049;
                }}
            </style>
        </head>
        <body>
            <h1>Prediction Result</h1>
            <h2>Survival Plot</h2>
            <img src="data:image/png;base64,{encoded_image}" alt="Cox Survival Plot">
            <a href="/">Go back</a>
        </body>
        </html>
        '''
    except ValueError as ve:
        return f"ValueError: Invalid input for age or sex. Details: {str(ve)}"
    except KeyError as ke:
        return f"KeyError: Missing input field. Details: {str(ke)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5050)

