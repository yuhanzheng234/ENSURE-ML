import pickle
import matplotlib.pyplot as plt
import base64
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive backend
import matplotlib.pyplot as plt
import io


def cox_model(df, clinical_endpoint):

    if clinical_endpoint == 'DFS':

        with open("/Users/mirandazheng/Desktop/first/ensure/medical_calculator/cox_dfs.pkl", "rb") as file:
            loaded_cox = pickle.load(file)

        surv_func = loaded_cox.predict_survival_function(df)
        ax = surv_func.iloc[:,:].plot() # attributes of pandas.dataframe.plot
        ax.legend().remove()
        plt.title(f'Cox predicted survival functions for {clinical_endpoint}')
        plt.xlabel('Time (months)')
        plt.ylabel('Survival probability')
        plt.grid()


    if clinical_endpoint == 'OS':

        with open("/Users/mirandazheng/Desktop/first/ensure/medical_calculator/cox_os.pkl", "rb") as file:
            loaded_cox = pickle.load(file)

        surv_func = loaded_cox.predict_survival_function(df)
        ax = surv_func.iloc[:,:].plot() # attributes of pandas.dataframe.plot
        ax.legend().remove()
        plt.title(f'Cox predicted survival functions for {clinical_endpoint}')
        plt.xlabel('Time (months)')
        plt.ylabel('Survival probability')
        plt.grid()



    if clinical_endpoint == 'DSS':

        with open("/Users/mirandazheng/Desktop/first/ensure/medical_calculator/cox_dss.pkl", "rb") as file:
            loaded_cox = pickle.load(file)

        surv_func = loaded_cox.predict_survival_function(df)
        ax = surv_func.iloc[:,:].plot() # attributes of pandas.dataframe.plot
        ax.legend().remove()
        plt.title(f'Cox predicted survival functions for {clinical_endpoint}')
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

