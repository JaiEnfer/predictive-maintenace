from pm.data.load_data import load_training_data
from pm.features.build_features import add_rul, add_binary_label
from pm.models.split import split_by_unit


def test_split_by_unit_no_overlap():
    df = load_training_data("FD001")
    df = add_rul(df)
    df = add_binary_label(df, threshold=30)

    train_df, val_df = split_by_unit(df, test_size=0.2, random_state=42)

    train_units = set(train_df["unit_number"].unique())
    val_units = set(val_df["unit_number"].unique())

    assert train_units.isdisjoint(val_units)
    assert len(train_units) > 0
    assert len(val_units) > 0
