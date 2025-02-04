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
        <title>Medical Calculator</title>
    </head>
    <body>
        <h1>Medical Calculator</h1>
        <form action="/predict" method="post">

            <label for="Sex">Sex (0 for Female, 1 for Male, NA):</label>
            <input type="text" id="Sex" name="Sex" required><br><br>

            <label for="Age at diagnosis">Age at diagnosis (float or NA):</label>
            <input type="text" id="Age at diagnosis" name="Age at diagnosis" required><br><br>

            <label for="ASA grade">ASA grade (1, 2, 3, NA):</label>
            <input type="text" id="ASA grade" name="ASA grade" required><br><br>

            <label for="Barrett's esophagus">Barrett's esophagus (0, 1, NA):</label>
            <input type="text" id="Barrett's esophagus" name="Barrett's esophagus" required><br><br>

            <label for="Histologic type">Histologic type (0, 1, NA):</label>
            <input type="text" id="Histologic type" name="Histologic type" required><br><br>

            <label for="Clinical T stage">Clinical T stage (0, 1, 2, 3, 4, NA):</label>
            <input type="text" id="Clinical T stage" name="Clinical T stage" required><br><br>

            <label for="Clinical N stage">Clinical N stage (0, 1, 2, 3, NA):</label>
            <input type="text" id="Clinical N stage" name="Clinical N stage" required><br><br>

            <label for="Clinical M stage">Clinical M stage (0, 1, NA):</label>
            <input type="text" id="Clinical M stage" name="Clinical M stage" required><br><br>

            <label for="Clinical Differentiation">Clinical Differentiation (0, 1, 2, 3, 4, NA):</label>
            <input type="text" id="Clinical Differentiation" name="Clinical Differentiation" required><br><br>

            <label for="Tumour site">Tumour site (0, 1, 2, 3, NA):</label>
            <input type="text" id="Tumour site" name="Tumour site" required><br><br>

            <label for="Distal margin positive">Distal margin positive (0, 1, NA):</label>
            <input type="text" id="Distal margin positive" name="Distal margin positive" required><br><br>

            <label for="Proximal margin positive">Proximal margin positive (0, 1, NA):</label>
            <input type="text" id="Proximal margin positive" name="Proximal margin positive" required><br><br>

            <label for="Radial margin positive">Radial margin positive (0, 1, NA):</label>
            <input type="text" id="Radial margin positive" name="Radial margin positive" required><br><br>

            <label for="Pathologic T stage">Pathologic T stage (0, 1, 2, 3, 4, 5, NA):</label>
            <input type="text" id="Pathologic T stage" name="Pathologic T stage" required><br><br>

            <label for="Pathologic N stage">Pathologic N stage (0, 1, 2, 3, NA):</label>
            <input type="text" id="Pathologic N stage" name="Pathologic N stage" required><br><br>
            
            <label for="Pathologic M Stage">Pathologic M Stage (0, 1, NA):</label>
            <input type="text" id="Pathologic M Stage" name="Pathologic M Stage" required><br><br>

            <label for="Differentiation">Differentiation (0, 1, 2, 3, 4, NA):</label>
            <input type="text" id="Differentiation" name="Differentiation" required><br><br>

            <label for="Lymphatic invasion">Lymphatic invasion (0, 1, NA):</label>
            <input type="text" id="Lymphatic invasion" name="Lymphatic invasion" required><br><br>

            <label for="Venous invasion">Venous invasion (0, 1, NA):</label>
            <input type="text" id="Venous invasion" name="Venous invasion" required><br><br>

            <label for="Perineural invasion">Perineural invasion (0, 1, NA):</label>
            <input type="text" id="Perineural invasion" name="Perineural invasion" required><br><br>

            <label for="Number of nodes analyzed">Number of nodes analyzed (integer or NA):</label>
            <input type="text" id="Number of nodes analyzed" name="Number of nodes analyzed" required><br><br>

            <label for="Treatment protocol">Treatment protocol (0, 1, 2, 3, 4, NA):</label>
            <input type="text" id="Treatment protocol" name="Treatment protocol" required><br><br>

            <label for="Operation type">Operation type (0, 1, 2, 3, 4, 5, 6, NA):</label>
            <input type="text" id="Operation type" name="Operation type" required><br><br>

            <label for="Approach">Approach (0, 1, 2, 3, 4, NA):</label>
            <input type="text" id="Approach" name="Approach" required><br><br>

            <label for="Robotic assisted">Robotic assisted (0, 1, NA):</label>
            <input type="text" id="Robotic assisted" name="Robotic assisted" required><br><br>

            <label for="Clavien-Dindo Grade">Clavien-Dindo Grade (0, 1, 2, 3, 4, 5, 6, 7, NA):</label>
            <input type="text" id="Clavien-Dindo Grade" name="Clavien-Dindo Grade" required><br><br>

            <label for="Length of stay">Length of stay (integer or NA):</label>
            <input type="text" id="Length of stay" name="Length of stay" required><br><br>

            <label for="Gastroenteric leak">Gastroenteric leak (0, 1, NA):</label>
            <input type="text" id="Gastroenteric leak" name="Gastroenteric leak" required><br><br>

            <label for="GDP USD per cap">GDP USD per cap (float or NA):</label>
            <input type="text" id="GDP USD per cap" name="GDP USD per cap" required><br><br>

            <label for="H volutary">H volutary (float or NA):</label>
            <input type="text" id="H volutary" name="H volutary" required><br><br>

            <label for="HE Compulsory">HE Compulsory (float or NA):</label>
            <input type="text" id="HE Compulsory" name="HE Compulsory" required><br><br>

            <label for="Cancer cases per year">Cancer cases per year (0, 1, 2, 3, 4, 5, 6, NA):</label>
            <input type="text" id="Cancer cases per year" name="Cancer cases per year" required><br><br>

            <label for="Consultant Volume SPSS">Consultant Volume SPSS (0, 1, 2, NA):</label>
            <input type="text" id="Consultant Volume SPSS" name="Consultant Volume SPSS" required><br><br>

            <label for="Intensive Surveillance">Intensive Surveillance (0, 1, NA):</label>
            <input type="text" id="Intensive Surveillance" name="Intensive Surveillance" required><br><br>

            <label for="Clinical Endpoint">Clinical Endpoint (DFS, OS, DSS):</label>
            <input type="text" id="Clinical Endpoint" name="Clinical Endpoint" required><br><br>

            <label for="Survival Model">Survival Model (CoxPH, XGBoost, DeepSurv, DeepHit):</label>
            <input type="text" id="Survival Model" name="Survival Model" required><br><br>



            <button type="submit">Calculate</button>
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

