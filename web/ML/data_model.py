import pandas as pd
import zipfile, os

fake_zip_path = 'web/ML/csv_train/Fake.csv.zip'
real_zip_path = 'web/ML/csv_train/True.csv.zip'
output_zip_path = 'web/ML/csv_train/Combined.zip'
output_csv_name = 'web/ML/csv_train/Combined.csv'

def extract_csv_from_zip(zip_path, extract_to='web/ML/csv_train'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('.csv'):
                zip_ref.extract(file_name, extract_to)
                return os.path.join(extract_to, file_name)
    return None


def merge_and_process():
    def preprocess(df):
        df = df.drop(columns=['date', 'subject'])
        df = df.drop_duplicates()
        return df
    
    fake_csv_file = extract_csv_from_zip(fake_zip_path)
    real_csv_file = extract_csv_from_zip(real_zip_path)
    
    fake_pd = pd.read_csv(fake_csv_file) # type: ignore
    real_pd = pd.read_csv(real_csv_file) # type: ignore
    
    fake_pd = preprocess(fake_pd)
    real_pd = preprocess(real_pd)
    
    fake_pd['Label'] = 0
    real_pd['Label'] = 1
    
    merged_df = pd.concat([fake_pd, real_pd], ignore_index='true') # type: ignore
    
    merged_df.to_csv(output_csv_name, index=False)
    
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_csv_name, arcname='Combined.csv')
        
    os.remove(fake_csv_file) # type: ignore
    os.remove(real_csv_file) # type: ignore
    os.remove(output_csv_name)



merge_and_process()
extract_csv_from_zip(output_zip_path)