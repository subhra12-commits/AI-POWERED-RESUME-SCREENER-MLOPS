import pandas as pd

def load_dataset(path=r"E:\resume-screener-mlops\archive\UpdatedResumeDataSet.csv"):
    df = pd.read_csv(path, encoding='latin1')  # or 'ISO-8859-1'


    df = df.drop_duplicates(subset='Resume')  # remove duplicate resumes

    return df
