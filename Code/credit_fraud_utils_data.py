import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def prepare_and_split_data(df, target_col = 'Class', test_size=0.15, val_size=0.15, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=adjusted_val_size, random_state=random_state, stratify=y_temp)

    scaler = RobustScaler()

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    scale_cols = ['Time', 'Amount']
    X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_val_scaled[scale_cols] = scaler.transform(X_val[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


if __name__ == '__main__':
    df = load_data(r'..\Data\creditcard.csv')
    X_tr, X_v, X_te, y_tr, y_v, y_te, scaler = prepare_and_split_data(df)
    print(f'Data Utiles Working Perfectly!')
    print(f'Train Shape: X = {X_tr.shape}, y = {y_tr.shape}')
    print(f'Val   Shape: X = {X_v.shape}, y = {y_v.shape}')
    print(f'test  Shape: X = {X_te.shape}, y = {y_te.shape}')