import pandas as pd
for p in ['data_processed/final_covid_dataset.csv','data_processed/feature_engineered_dataset.csv']:
    print('\nChecking', p)
    df = pd.read_csv(p)
    print('rows', df.shape[0], 'cols', df.shape[1])
    print(df.head())
