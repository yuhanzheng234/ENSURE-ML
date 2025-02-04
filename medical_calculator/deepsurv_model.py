import matplotlib.pyplot as plt
from sklearn_pandas import DataFrameMapper
import pickle
import base64
import io



def deepsurv_model(df, clinical_endpoint):


    if clinical_endpoint == 'DFS':

        with open("/Users/mirandazheng/Desktop/first/ensure/medical_calculator/deepsurv_dfs.pkl", "rb") as file:
            loaded_deepsurv = pickle.load(file)

        cols_leave = df.columns
        leave = [(col, None) for col in cols_leave]
        x_mapper = DataFrameMapper(leave)
        x_test = x_mapper.fit_transform(df).astype('float32') 

        surv = loaded_deepsurv.predict_surv_df(x_test)
        ax = surv.iloc[:,:].plot() # attributes of pandas.dataframe.plot
        ax.legend().remove()
        plt.title(f'DeepSurv predicted survival functions for {clinical_endpoint}')
        plt.xlabel('Time (months)')
        plt.ylabel('Survival probability')
        plt.grid()


    if clinical_endpoint == 'OS':
        with open("/Users/mirandazheng/Desktop/first/ensure/medical_calculator/deepsurv_os.pkl", "rb") as file:
            loaded_deepsurv = pickle.load(file)

        cols_leave = df.columns
        leave = [(col, None) for col in cols_leave]
        x_mapper = DataFrameMapper(leave)
        x_test = x_mapper.fit_transform(df).astype('float32') 

        surv = loaded_deepsurv.predict_surv_df(x_test)
        ax = surv.iloc[:,:].plot() # attributes of pandas.dataframe.plot
        ax.legend().remove()
        plt.title(f'DeepSurv predicted survival functions for {clinical_endpoint}')
        plt.xlabel('Time (months)')
        plt.ylabel('Survival probability')
        plt.grid()


    if clinical_endpoint == 'DSS':

        with open("/Users/mirandazheng/Desktop/first/ensure/medical_calculator/deepsurv_dss.pkl", "rb") as file:
            loaded_deepsurv = pickle.load(file)

        cols_leave = df.columns
        leave = [(col, None) for col in cols_leave]
        x_mapper = DataFrameMapper(leave)
        x_test = x_mapper.fit_transform(df).astype('float32') 

        surv = loaded_deepsurv.predict_surv_df(x_test)
        ax = surv.iloc[:,:].plot() # attributes of pandas.dataframe.plot
        ax.legend().remove()
        plt.title(f'DeepSurv predicted survival functions for {clinical_endpoint}')
        plt.xlabel('Time (months)')
        plt.ylabel('Survival probability')
        plt.grid()

    # Save the plot to a BytesIO buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Encode the image to base64
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    return encoded_image