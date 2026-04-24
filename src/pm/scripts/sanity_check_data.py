from pm.data.load_data import load_training_data, load_test_data, load_rul

def main():
    train = load_training_data("FD001")
    test = load_test_data("FD001")
    rul = load_rul("FD001")

    print("Train Shape: ", train.shape)
    print("Test Shape: ", test.shape)
    print("RUL Length: ", len(rul))

    print("\nTRAIN head:\n", train.head())
    print("\nColumns:\n", list(train.columns))

if __name__ == "__main__":
    main()

