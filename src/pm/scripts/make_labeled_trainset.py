from pm.data.load_data import load_training_data
from pm.features.build_features import add_rul, add_binary_label


def main():
    df = load_training_data("FD001")
    df = add_rul(df)
    df = add_binary_label(df, threshold=30)

    print("Shape:", df.shape)
    print("RUL min/max:", df["RUL"].min(), df["RUL"].max())
    print("Label counts:\n", df["will_fail_soon"].value_counts())

    # sanity: show last few cycles of engine 1
    e1 = df[df["unit_number"] == 1].sort_values("time_in_cycles").tail(5)
    print("\nEngine 1 tail:\n", e1[["unit_number", "time_in_cycles", "RUL", "will_fail_soon"]])


if __name__ == "__main__":
    main()
