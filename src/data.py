from sklearn.datasets import fetch_california_housing

def load_california_housing():
    """
    Load the California Housing dataset as a pandas DataFrame.

    Returns:
            pd.DataFrame: California Housing dataset.
    """
    
    housing = fetch_california_housing(
        as_frame=True)

    return housing.frame