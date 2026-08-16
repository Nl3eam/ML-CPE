from sklearn.model_selection import train_test_split


def split_dataset(X, y, test_size=0.2):
    # Works the same whether X is a DataFrame (our case) or an array,
    # and y must support stratify either way.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
