import os

import pandas as pd
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error

import kerastuner as kt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import Callback, EarlyStopping


#1 Model XGBregressor

#We are going to try out differnet n_estimators and learning
#rates to find the best hyperparameters
def xgb_model(n_estimators=[], learning_rate=[], validation_data=(), training_data=(), testing_data=(), directory='', filename=''):
    '''
    Takes a list of estimators and learning rate
    along with train/valid/test data.
    
    Runs the XGB regressor saves
    the weights in .model format and
    the performances in a csv file 
    and returns the performance
    results back in a dataFrame
    '''
    mse = {}
    for estimator in n_estimators:
        for rate in learning_rate:
            #Inisiating the model
            model = XGBRegressor(n_estimators=estimator, 
                                 learning_rate=rate)
            
            #Training the model
            model.fit(training_data[0], training_data[1],
                     early_stopping_rounds=50,
                     eval_set = [(validation_data[0], validation_data[1])],
                     verbose=False)
            
            #Evaluating the model
            prediction = model.predict(testing_data[0])
            
            #saving the model
            model.save_model('../src/models/xgb_weights/n_estimator{}_learning_rate{}.model'.format(estimator, rate))
            
            #Calculating the error
            error = mean_squared_error(prediction, testing_data[1])
            mse[error] = [estimator, rate]
    
    #Converting the dict to a DataFrame
    xgb_performance = pd.DataFrame(data=mse)
    xgb_performance = xgb_performance.transpose()
    xgb_performance.columns = ['n_estimator', 'learning_rate']
    xgb_performance.index.name = 'mse'
    
    #Saving the performances in a CSV file
    if os.path.exists(directory):
        xgb_performance.to_csv('../src/models/{}/{}.csv'.format(directory,filename))
    else:
        os.makedirs(directory)
        xgb_performance.to_csv('../src/models/{}/{}.csv'.format(directory,filename))
    return xgb_performance

#2nd model neural networks
def model_build(hp):
    model = Sequential()
    input_shape = (27,)
    
    #Activation function and neural units to choose from
    hp_units = hp.Int('units', min_value = 4,max_value = 28,step = 4, default=16)
    hp_choice = hp.Choice('dense_activation',
                          values=['relu','elu'],
                            default='relu')
    #For Dropout layer
    hp_float = hp.Float('dropout',
                       min_value=0.0,
                       max_value=0.3,
                       default=0.15,
                       step=0.01)

    model.add(Dense(units= hp_units,
                   activation =hp_choice,
                    input_shape = input_shape))


    model.add(Dense(units= hp_units,
                   activation =hp_choice))

    model.add(Dropout(hp_float))

    model.add(Dense(1))

    #Learning rate
    hp_learning_rate = hp.Choice('learning_rate', values= [0.01, 0.001, 0.001])

    model.compile(optimizer=RMSprop(learning_rate=hp_learning_rate),
                 loss = 'mse',
                 metrics=['mse'])

    return model

#For Display
import IPython
class ClearTrainingOutput(Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


def hyper_parameter_search(search_type='BO',objective='mse', seed=101, 
                           max_trails=10, directory=os.path.normpath('C:/'), 
                           project_name='', max_epochs=10, factor=3, epochs=10,
                          train_data=(), val_data=()):
    '''
    Given the search type this method uses that optimization 
    method from keras tuner and finds the best parameters.
    and returns the model with the best parameteres. 
    '''
    search_type = search_type.upper()
    
    if search_type == 'BO' or search_type == 'BAYESIANOPTIMIZATION':
        tuner = kt.BayesianOptimization(model_build,
                                  objective=objective,
                                        seed=seed,
                                  max_trials=max_trails,
                                   directory=directory,
                                  project_name=project_name)
    
    elif search_type == 'RS' or search_type == 'RANDOMSEARCH':
        tuner = kt.RandomSearch(model_build,
                          objective=objective,
                          seed=seed,
                          max_trials=max_trails,
                          directory=directory,
                            project_name = project_name)
    
    elif search_type == 'HB' or search_type == 'HYPERBAND':
        tuner = kt.Hyperband(model_build,
                     max_epochs=max_epochs,
                       objective=objective,
                       factor=factor,
                     directory=directory,
                    project_name=project_name)
    else:
        raise ValueError('The requested keras tuner search type doesnot exist\n')
    
    tuner.search(train_data[0], train_data[1], epochs=epochs, 
               validation_data = (val_data[0], val_data[1]),
               callbacks = [ClearTrainingOutput()], verbose=1)
    
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    
    print(f"""
        The hyperparameter search is complete. The optimal units
        {best_hps.get('units')} and the optimal learning rate is 
        {best_hps.get('learning_rate')} and the optimal dropout
        {best_hps.get('dropout')} and the optimal activation
        {best_hps.get('dense_activation')}.""")
    model = tuner.hypermodel.build(best_hps)
    return model


def model_fit( model, train_data =(),val_data=(), monitor='mse', patience=3, verbose=2, epochs=50):
    '''
    Fits the model to the validation data.
    '''
    callback = EarlyStopping(monitor=monitor, patience=patience, verbose=verbose)
    history = model.fit(train_data[0], train_data[1], epochs=epochs,
                       validation_data = (val_data[0], val_data[1]),
                      callbacks = [callback], verbose=1)
    return history, model