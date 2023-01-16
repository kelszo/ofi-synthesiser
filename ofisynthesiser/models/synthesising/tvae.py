import pandas as pd
import sdv.tabular


def generate_data_tvae(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    epochs = 300
    batch_size = 500

    if debug:
        epochs = 3
        batch_size = 50

    model = sdv.tabular.TVAE(epochs=epochs, batch_size=batch_size)
    model.fit(data)

    return model.sample(num_rows=len(data))
