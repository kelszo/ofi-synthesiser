import pandas as pd

def mean_scalar(data: pd.Series, scalar_name: str, group = None) -> str:
    mean = round(data.mean())
    std = round(data.std())


    return f"{mean} ({std})"

def median_scalar(data: pd.Series, scalar_name: str, group = None) -> str:
    median = round(data.median())
    min_value = round(data.min())
    max_value = round(data.max())

    return f"{median} [{min_value}, {max_value}]"

def format_category(data: pd.Series, category_name: str, group = None) -> str:
    n = data.sum()
    p = round((n / len(data))*100)

    return f"{n} ({p}\\%)"

def format_missing(data: pd.Series, category_name: str, group = None) -> str:
    n = data.isna().sum()
    p = round((n / len(data)* 100))

    return f"{n} ({p}\\%)"

def format_tableone(data: pd.DataFrame, groupby:str, missing = True) -> str:
    groupby_values = data[groupby].unique()

    table1_dict = {x:[] for x in ["","Overall", *data["ofi"].unique()]}

    for column in data.columns:
        if column == groupby:
            continue

        column_dtype = data[column].dtype

        if column_dtype not in ["category", "object"]:
            table1_dict[""].append("\\textbf{" + column + "}")

            table1_dict["Overall"].append("")
            for groupby_value in groupby_values:
                table1_dict[groupby_value].append("")

            table1_dict[""].append("\hspace{3mm}Mean (SD)")
            table1_dict["Overall"].append(mean_scalar(data[column], column))
            for groupby_value in groupby_values:
                table1_dict[groupby_value].append(mean_scalar(data.loc[data[groupby] == groupby_value][column], column, group=groupby_value))

            table1_dict[""].append("\hspace{3mm}Median [Min, Max]")
            table1_dict["Overall"].append(median_scalar(data[column], column))
            for groupby_value in groupby_values:
                table1_dict[groupby_value].append(median_scalar(data.loc[data[groupby] == groupby_value][column], column, group=groupby_value))

            if missing:
                table1_dict[""].append("\hspace{3mm}Missing")

                table1_dict["Overall"].append(format_missing(data[column], column))
                
                for groupby_value in groupby_values:
                    table1_dict[groupby_value].append(format_missing(data.loc[data[groupby] == groupby_value][column], column, group=groupby_value))

        elif column_dtype == "category":
            table1_dict[""].append("\\textbf{" + column + "}")
            table1_dict["Overall"].append("")
            for groupby_value in groupby_values:
                table1_dict[groupby_value].append("")

            for category in data[column].dropna().unique():
                table1_dict[""].append("\hspace{3mm}" + str(category))
                
                table1_dict["Overall"].append(format_category(data[column] == category, column))

                for groupby_value in groupby_values:
                    table1_dict[groupby_value].append(format_category(data.loc[data[groupby] == groupby_value][column] == category, column, group=groupby_value))
            
            if missing:
                table1_dict[""].append("\hspace{3mm}Missing")

                table1_dict["Overall"].append(format_missing(data[column], column))
                
                for groupby_value in groupby_values:
                    table1_dict[groupby_value].append(format_missing(data.loc[data[groupby] == groupby_value][column], column, group=groupby_value))
            


    return pd.DataFrame(table1_dict)