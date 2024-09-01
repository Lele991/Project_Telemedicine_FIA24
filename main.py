import pandas as pd
from dataprocessing.ManageData import ManageData
from dataprocessing.manage import DataFix, DataCleaner

def load_data(file_path):
    # Load the data
    df = pd.read_parquet(file_path)
    return df


def main():
    # Specify the path to the Parquet file
    file_path_data = 'data/challenge_campus_biomedico_2024.parquet'
    file_path_province = 'data/italia/province.json'
    file_path_comuni = 'data/italia/comuni.json'

    # Load the Parquet file into a DataFrame
    df = load_data(file_path_data)

    # Print the DataFrame
    #print(df)

    data = ManageData(df, file_path_province, file_path_comuni)
    data.replace_none_with_nan()
    data.fix_province()

if __name__ == "__main__":
    main()