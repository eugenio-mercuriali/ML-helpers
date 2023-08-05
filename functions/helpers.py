import pandas as pd
import numpy as np
from typing import Union
from sklearn.base import ClassifierMixin


class MissingInputError(Exception):
    pass


def smart_undersample(
    model: ClassifierMixin,
    df_train: Union[pd.DataFrame, None] = None,
    X_train: Union[pd.DataFrame, None] = None,
    y_train: Union[pd.Series, None] = None,
    n_runs: int = 30
) -> pd.DataFrame:
    """
    Bootstrapped hard negative sampling.
    Solving a class-imbalance case (with the negative class as the majority class)
    by undersampling the negative class by keeping only top k hard negatives samples.

    :param model: The classifier model to be used for bootstrapped predictions.
    :param df_train: The training dataframe containing features and target.
                     Either df_train or X_train and y_train need to be passed.
                     Default is None.
    :param X_train: The training features dataframe.
                    Required if df_train is None.
                    Default is None.
    :param y_train: The training target series.
                    Required if df_train is None.
                    Default is None.
    :param n_runs: The number of bootstrap runs. Default is 30.

    :return: The undersampled training dataframe after bootstrapped predictions.
    :rtype: pd.DataFrame
    """

    # Check if either df_train or X_train and y_train are provided
    if (X_train is None or y_train is None) and df_train is None:
        raise MissingInputError('Either df_train or X_train and y_train needs to be passed')

    # Convert X_train and y_train to df_train if only X_train and y_train are provided
    if (X_train is not None or y_train is not None) and df_train is None:
        df_train = X_train.copy()
        df_train['y'] = y_train

    # Convert df_train to X_train and y_train if only df_train is provided
    elif (X_train is None or y_train is None) and df_train is not None:
        X_train = df_train.copy()
        X_train = X_train.drop('y', axis=1)
        y_train = df_train['y']

    # Get the unique classes to predict
    classes = y_train.unique()

    # Creating a dictionary to store the bootstrapped prediction probabilities
    bootstrapped_predictions = {f'class_{x}': [0] * len(y_train) for x in classes}

    # Sample n negatives, where n is the number of positives
    n_obs = len(y_train[y_train != 0])
    positive_index_list = y_train[y_train != 0].index

    for i in range(n_runs):
        # List of sampled negatives indices
        negative_index_list = df_train[df_train['y'] == 0].sample(n_obs, random_state=i).index

        # Fit the model only on the sampled data
        model.fit(
            X_train[X_train.index.isin(negative_index_list + positive_index_list)],
            y_train[y_train.index.isin(negative_index_list + positive_index_list)]
        )

        # Predict probabilities for all the observations
        pred_probabilities = model.predict_proba(X_train)

        # Update predictions for all observations
        for class_num in classes:
            bootstrapped_predictions[f'class_{class_num}'] = np.add(
                bootstrapped_predictions[f'class_{class_num}'],
                pred_probabilities[:, class_num]
            )

    # Compute the average prediction probabilities over n_runs
    average_pred_probabilities = {
        f'class_{x}': np.divide(bootstrapped_predictions[f'class_{x}'], n_runs)
        for x in classes
    }

    # Total number of positives
    n_positives = len(df_train[df_train['y'] != 0])

    # Add the average prediction probabilities to the original DataFrame
    for key in average_pred_probabilities.keys():
        df_train[f'{key}_prob'] = average_pred_probabilities[key]

    # Rank negatives to extract hard negatives
    df_train_undersampled = (
        df_train[df_train['y'] == 0]
        .sort_values(by='class_0_prob', ascending=True)
        .iloc[:n_positives, :]
    )

    # Add positives
    df_train_undersampled = pd.concat(
        [df_train_undersampled, df_train[df_train['y'] != 0]],
        axis=0
    )

    # Return the undersampled DataFrame without the extra columns for classes
    return df_train_undersampled.iloc[:, :-len(y_train.unique())]
