import pandas as pd
#Getting rid of jobId and company Id and making dummies
# for jobType, degree, major, industry.

def pre_processing(data):
    '''
    Gets rid of the jobId and company Id and makes dummies
     for jobType, degree, major, industry.
    '''
    data = data.drop(labels=['companyId', 'jobId'], axis=1)
    data = pd.get_dummies(data=data, columns=['jobType', 
                                                'degree', 
                                                'major', 
                                                'industry'], drop_first=True)
    return data

def preprocessing_pipeline(data, scaler):
    if 'companyId' and 'jobId' in data:
        data = pre_processing(data=data)
    data[['yearsExperience', 'milesFromMetropolis']] = scaler.transform(data[['yearsExperience', 'milesFromMetropolis']])
    return data