import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset


# data splitting function that takes in 2D array X and 1D array y and returns 2 dictionaries
def data_split(X,y,nan_mask,indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }

    if x_d['data'].shape != x_d['mask'].shape:
        raise'Shape of data not same as that of nan mask!'

    y_d = {
        'data': y[indices].reshape(-1, 1)
    }
    return x_d, y_d


# Preparing X (input) and y (ground truths) dataset
def data_prep(X_pre_norm, y, seed = 42, datasplit=[.65, .15, .2]):

    np.random.seed(seed)
    X = pd.DataFrame(X_pre_norm)

    categorical_indicator = [False, False, False, False, False]
    categorical_columns = X.columns[list(np.where(np.array(categorical_indicator) == True)[0])].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))


    cat_idxs = list(np.where(np.array(categorical_indicator) == True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

    for col in categorical_columns:
        X[col] = X[col].astype("object")

    X["Set"] = np.random.choice(
        ["train", "valid", "test"], p=datasplit, size=(X.shape[0], ))  # randomly categorise rows in the dataframe X into the newly created Set column as training, validation and testing.

    train_indices = X[X.Set == "train"].index  # contains indices of rows that are assigned as training data
    valid_indices = X[X.Set == "valid"].index  # contains indices of rows that are assigned as validation data
    test_indices = X[X.Set == "test"].index  # contains indices of rows that are assigned as testing data

    X = X.drop(columns=['Set'])
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)

    cat_dims = []
    for col in categorical_columns:
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder()
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))

    for col in cont_columns:
        X.fillna(X.loc[train_indices, col].mean(), inplace=True)

    #X = X[cont_columns].copy()
    # X = (X-X.mean())/X.std()
    #X.values = (X.values-X.values.mean())/X.values.std()

    X_train, y_train = data_split(X, y, nan_mask, train_indices)
    X_valid, y_valid = data_split(X, y, nan_mask, valid_indices)
    X_test, y_test = data_split(X, y, nan_mask, test_indices)

    train_mean, train_std = np.array(X_train['data'][:, con_idxs],
                                     dtype=np.float32).mean(0), np.array(
                                         X_train['data'][:, con_idxs],
                                         dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    # import ipdb; ipdb.set_trace()
    # Testing Outputs:
    # return train_indices, valid_indices, test_indices
    #return X_train, y_train, X_valid, y_valid, X_test, y_test, nan_mask

    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std


# this class of functions is used to handle the heterogenous data in a structured manner
class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols,continuous_mean_std=None):

        cat_cols = list(cat_cols)
        X_mask =  X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        self.y = Y['data'].astype(np.float32)
        self.cls = np.zeros_like(self.y,dtype=int)
        self.cls_mask = np.ones_like(self.y,dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx],self.y[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]