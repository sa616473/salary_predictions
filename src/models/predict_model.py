import pandas as pd
from tensorflow.keras.models import load_model

def making_predictions(data, jobId):
    model = load_model('../src/models/neuralnetowrk_weights/model_bo.h5')
    predictions = model.predict(data)
    
    predictions = pd.DataFrame(predictions)
    predictions.columns = ['predicted_salary']
    predictions = pd.concat([jobId, predictions], axis=1)
    predictions.to_csv('predicted_salary.csv')
    return predictions