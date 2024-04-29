import numpy as np # linear algebra
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler, StandardScaler, OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import FunctionTransformer
import xgboost
import pickle
from datetime import datetime


fecha_hoy = datetime.now().strftime('%Y-%m-%d')
cat_features = ['manufacturer','transmission','drivetrain','fuel_type']
num_features = ['year','mileage', 'mpg', 'driver_reviews_num', 'seller_rating', 'driver_rating']
dum_variables = ['accidents_or_damage', 'one_owner','personal_use_only']
only_features = dum_variables + cat_features + num_features

dive_train_map = {'Front-wheel Drive':'Front Wheel Driver',
                    'FWD':'Front Wheel Driver',
                    'Front-Wheel Drive':'Front Wheel Driver',
                    'Four-wheel Drive':'Four Wheel Driver',
                    'Four-Wheel Drive':'Four Wheel Driver',
                    'Four Wheel Drive':'Four Wheel Driver',
                    '4WD':'Four Wheel Driver',
                    'All-wheel Drive':'All Wheel Driver',
                    'All-Wheel Drive':'All Wheel Driver',
                    'AWD':'All Wheel Driver',
                    'Rear-wheel Drive':'Rear Wheel Driver',
                    'RWD':'Rear Wheel Driver'}


def get_mpg(x):
    x = str(x)
    if x == 'nan':
        return np.nan
    elif len(x) <= 2:
        return float(x)
    else:
        return (float(x.split('-')[0])+float(x.split('-')[1]))/2
    

def proceso_princial():
    # Directorio donde se encuentra el archivo
    path_for_file = 'archivos_a_procesar/data_cars_new.csv'
    df = pd.read_csv(path_for_file)
    qty_reg_file = len(df)
    
    # Quito los nulos
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)

    # Transformaciones necesarias   
    for i in dum_variables:
        df[i] = df[i].astype(int)

    df['drivetrain'] = df['drivetrain'].map(dive_train_map).fillna(value='Others')
    df['mpg'] = df['mpg'].map(get_mpg)

    with open('models/xgb_model_v1.pkl', 'rb') as file:
        model = pickle.load(file)

    data_prediction = df[only_features].copy()
    qty_reg_results = len(data_prediction)

    # Predicciones
    predicciones = model.predict(data_prediction)
    df_predict = pd.DataFrame({'estimacion_precio':predicciones})

    # Guardado archivo
    resultado = pd.concat([df[['manufacturer','model','year','transmission','drivetrain']],df_predict], axis=1)
    resultado.to_csv(f'resultados_predicciones/predicciones_{fecha_hoy}.csv', index=False)
    print(f'Cantidad de registros input : {qty_reg_file}')
    print(f'Cantidad de registros procesados: {qty_reg_results}')


if __name__ == '__main__':
    proceso_princial()