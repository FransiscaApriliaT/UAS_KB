def split_data(features, labels, test_size=0.3, random_state=42):
    from sklearn.model_selection import train_test_split
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)
