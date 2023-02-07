import pandas as pd
import sdv.tabular
import sdv.sampling


def generate_data_copula_gan(data: pd.DataFrame, cuda: bool, debug: bool = False) -> pd.DataFrame:
    epochs = 600
    batch_size = 1000

    if debug:
        epochs = 1
        batch_size = 500

    model = sdv.tabular.CopulaGAN(epochs=epochs, cuda=cuda, batch_size=batch_size, verbose=debug)
    model.fit(data)

    df_synth = model.sample(num_rows=len(data))

    if len(df_synth["ofi"].value_counts()) == 1:
        condition = sdv.sampling.Condition({"ofi": 1}, num_rows=data["ofi"].value_counts()[1])
        df_ofi_synth = model.sample_conditions(conditions=[condition])

        df_synth = pd.concat([df_synth, df_ofi_synth])

    return df_synth
