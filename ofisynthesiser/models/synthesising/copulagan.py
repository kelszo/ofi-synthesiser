import pandas as pd
import sdv.tabular


def generate_data_copula_gan(data: pd.DataFrame, cuda: bool, debug: bool = False) -> pd.DataFrame:
    epochs = 300
    batch_size = 500

    if debug:
        epochs = 3
        batch_size = 50

    model = sdv.tabular.CopulaGAN(epochs=epochs, cuda=cuda, batch_size=batch_size)
    model.fit(data)

    return model.sample(num_rows=len(data))
