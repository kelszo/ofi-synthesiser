import pandas as pd
import sdv.tabular


def generate_data_tvae(data: pd.DataFrame, cuda: bool, debug: bool = False) -> pd.DataFrame:
    epochs = 500
    batch_size = 2000

    if debug:
        epochs = 1
        batch_size = 50

    model = sdv.tabular.TVAE(epochs=epochs, cuda=cuda, batch_size=batch_size)
    model.fit(data)

    return model.sample(num_rows=len(data))
