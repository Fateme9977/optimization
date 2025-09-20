import faiss
import numpy as np
import pandas as pd

class AnalogDayRetriever:
    """
    Retrieves the top-K most similar historical days using a FAISS index.
    """
    def __init__(self, historical_data: pd.DataFrame, feature_cols: list):
        """
        Args:
            historical_data (pd.DataFrame): DataFrame containing historical data.
                                            Must have a DatetimeIndex.
            feature_cols (list): List of column names to use for building the index.
        """
        if not isinstance(historical_data.index, pd.DatetimeIndex):
            raise ValueError("historical_data must have a DatetimeIndex.")

        self.feature_cols = feature_cols
        # We assume one row per day. If data is finer, it needs to be aggregated.
        # For this example, we'll resample to daily averages.
        self.daily_data = historical_data[feature_cols].resample('D').mean().dropna()
        self.index = None

    def build_index(self):
        """Builds the FAISS index from the historical data."""
        print("Building FAISS index...")
        features = self.daily_data.values.astype('float32')

        # Normalize features for better distance calculation
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)
        features = (features - self.mean) / self.std

        d = features.shape[1]  # dimension of vectors
        self.index = faiss.IndexFlatL2(d)
        self.index.add(features)
        print(f"Index built successfully with {self.index.ntotal} vectors.")

    def find_analog_days(self, query_date: pd.Timestamp, k=5):
        """
        Finds the k most similar days to a given query date.

        Args:
            query_date (pd.Timestamp): The date to find analogs for.
            k (int): The number of similar days to retrieve.

        Returns:
            pd.DataFrame: A DataFrame containing the data for the k analog days.
        """
        if self.index is None:
            raise RuntimeError("Index has not been built yet. Call build_index() first.")

        try:
            query_vector = self.daily_data.loc[query_date.strftime('%Y-%m-%d')].values.astype('float32')
        except KeyError:
            print(f"Warning: No historical data for query date {query_date}. Cannot find analogs.")
            return pd.DataFrame()

        # Normalize the query vector using the same stats as the index
        query_vector = (query_vector - self.mean) / self.std
        query_vector = np.expand_dims(query_vector, axis=0)

        distances, indices = self.index.search(query_vector, k)

        analog_dates = self.daily_data.index[indices[0]]
        return self.daily_data.loc[analog_dates]

if __name__ == '__main__':
    # --- Example Usage ---
    # Create a dummy historical dataset
    date_rng = pd.to_datetime(pd.date_range(start='2022-01-01', end='2022-12-31', freq='D'))
    data = {
        'month': date_rng.month,
        'dayofweek': date_rng.dayofweek,
        'avg_temp': np.random.rand(len(date_rng)) * 30,
        'csi_profile': np.random.rand(len(date_rng))
    }
    historical_df = pd.DataFrame(data, index=date_rng)

    feature_columns = ['month', 'dayofweek', 'avg_temp', 'csi_profile']

    # 1. Initialize the retriever
    retriever = AnalogDayRetriever(historical_df, feature_cols=feature_columns)

    # 2. Build the index
    retriever.build_index()

    # 3. Find analog days for a query date
    query_day = pd.Timestamp('2022-07-15')
    print(f"\nFinding analog days for {query_day.date()}...")
    analog_days = retriever.find_analog_days(query_day, k=5)

    print("\n--- Top 5 Analog Days ---")
    print(analog_days)
    assert len(analog_days) == 5
    print("\nRetrieval logic verified.")
