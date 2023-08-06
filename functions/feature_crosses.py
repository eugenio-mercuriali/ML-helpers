import pandas as pd
from itertools import combinations

def create_feature_crosses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates feature crosses between numerical features of a pandas df
    and returns the enhanced dataframe
    :param df: pandas Dataframe
    :return: a pandas DataFrame containing the original features and the feature crosses
    """

    # Create a new DataFrame to store the feature crosses
    df_crosses = pd.DataFrame()

    # Generate all combinations of column pairs
    columns = df.columns.tolist()
    column_combinations = list(combinations(columns, 2))

    # Iterate through each combination of columns
    for column_pair in column_combinations:
        # Extract the column names
        column1, column2 = column_pair

        # Perform the feature cross by multiplying the values of the two columns
        feature_cross = df[column1] * df[column2]

        # Name the new feature cross column
        feature_cross_name = f'{column1}_x_{column2}'

        # Add the feature cross column to the new DataFrame
        df_crosses[feature_cross_name] = feature_cross

    # Include the squares of individual columns as feature crosses
    for column in columns:
        square_cross = df[column] * df[column]
        square_cross_name = f'{column}_x_{column}'
        df_crosses[square_cross_name] = square_cross

    # Concatenate the original DataFrame and the feature cross DataFrame
    df_enhanced = pd.concat([df, df_crosses], axis=1)

    return df_enhanced
