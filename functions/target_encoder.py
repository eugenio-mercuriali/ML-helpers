import pandas as pd
import category_encoders
from category_encoders import MEstimateEncoder
from functools import reduce
from sklearn.model_selection import KFold


class CrossFoldEncoder:
    """
    CrossFoldEncoder is a class that performs cross-validation encoding on categorical features.
    The encoder is fitted on the training data of each fold and then applied to the corresponding validation data.
    The encoded data from all folds is then concatenated to form the final output.
    :param encoder: An encoding class or function that takes in a pandas DataFrame and the name(s) of categorical column(s) and returns an encoded DataFrame.
    :type encoder: class or function
    :param kwargs: Keyword arguments for the encoder.
    :type kwargs: dict
    """

    def __init__(self, encoder, n_folds: int = 5, **kwargs):
        """
        Initializes a new instance of the CrossFoldEncoder class.
        :param encoder: An encoding class or function that takes in a pandas DataFrame and the name(s) of categorical column(s) and returns an encoded DataFrame.
        :type encoder: class or function
        :param kwargs: Keyword arguments for the encoder.
        :type kwargs: dict
        """
        self.cols_ = None
        self.fitted_encoders_ = None
        self.encoder_ = encoder
        self.kwargs_ = kwargs
        self.cv_ = KFold(n_splits=n_folds)

    def fit_transform(
            self, x: pd.DataFrame, y: pd.Series, cols
    ) -> pd.DataFrame:
        """
        Fits the encoder on the training data of each fold and applies it to the corresponding validation data.
        The encoded data from all folds is then concatenated to form the final output.
        :param x: The input data to be encoded.
        :type x: pandas DataFrame
        :param y: The target variable.
        :type y: pandas Series
        :param cols: The name(s) of the categorical column(s) to be encoded.
        :type cols: str or list of str
        :return: A pandas DataFrame containing the encoded data.
        :rtype: pandas DataFrame
        """
        # Initialize empty lists and variables to store the fitted encoders and the column names
        self.fitted_encoders_ = []
        self.cols_ = cols
        x_encoded = []

        # Loop through each fold in the cross-validation
        for idx_encode, idx_train in self.cv_.split(x):
            # Instantiate a new encoder for this fold
            fitted_encoder = self.encoder_(cols=cols, **self.kwargs_)
            # Fit the encoder on the training data of this fold
            fitted_encoder.fit(x.iloc[idx_encode, :], y.iloc[idx_encode])

            # Apply the fitted encoder to the validation data of this fold and append the result to the list of
            # encoded data
            x_encoded.append(fitted_encoder.transform(x.iloc[idx_train, :])[cols])
            # Store the fitted encoder for later use
            self.fitted_encoders_.append(fitted_encoder)

        # Concatenate the encoded data from all folds into a single DataFrame
        x_encoded = pd.concat(x_encoded)
        # Rename the columns to include the "_encoded" suffix
        x_encoded.columns = [name + '_encoded' for name in x_encoded.columns]
        # Return the encoded data
        return x_encoded

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the fitted encoder to the input data and returns the encoded data.
        :param x: The input data to be encoded.
        :type x: pandas DataFrame
        :return: A pandas DataFrame containing the encoded data.
        :rtype: pandas DataFrame
        """

        # Initialize an empty list to store the encoded data from each fold
        x_encoded_list = []
        # Loop through each fitted encoder stored from the fit_transform method
        for fitted_encoder in self.fitted_encoders_:
            # Apply the fitted encoder to the input data
            x_encoded = fitted_encoder.transform(x)
            # Append the encoded data from this fold to the list
            x_encoded_list.append(x_encoded[self.cols_])

        # Average the encoded data from all folds by adding them and dividing by the number of folds
        x_encoded = reduce(lambda s, y: s.add(y, fill_value=0), x_encoded_list) / len(
            x_encoded_list
        )
        # Rename the columns to include the "_encoded" suffix
        x_encoded.columns = [name + '_encoded' for name in x_encoded.columns]
        # Return the encoded data
        return x_encoded
