import numpy as np
import pandas as pd
import xarray as xr


def _stacked_ds_to_df(ds: xr.Dataset, indices: np.ndarray) -> pd.DataFrame:
    """Create a pandas dataframe for ML purposes from a stacked Dataset."""
    df = ds.where(indices).dropna(dim="latlon").to_dataframe().dropna()
    df = df.drop(["latitude", "longitude"], axis=1)
    return df.reset_index()


def create_train_test_dataframes(
    ds: xr.Dataset, test_size: float, random_seed: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create training and testing dataframes, with reproducable random seed.

    Args:
        ds: Source dataset
        test_size: Size of the test data (fraction, 0.0 - 1.0)
        random_seed: Which random seed value to use for the train-test splitting.
            Defaults to None.

    Returns:
        training dataframe, testing dataframe
    """
    stacked = ds.stack(latlon=["latitude", "longitude"]).dropna(dim="latlon")

    if random_seed is not None:
        np.random.seed(random_seed)
    random_data = np.random.rand(stacked["latlon"].size)

    training_indices = random_data <= np.percentile(random_data, 100 * (1 - test_size))
    test_indices = (1 - training_indices).astype(bool)

    df_train = _stacked_ds_to_df(stacked, training_indices)
    df_test = _stacked_ds_to_df(stacked, test_indices)

    return df_train, df_test
