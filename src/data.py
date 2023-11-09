import h5py
import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms


def load_mnist() -> tuple:
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root='../data',
                               train=True,
                               download=True,
                               transform=tensor_transform)
    test_set = datasets.MNIST(root='../data',
                              train=False,
                              download=True,
                              transform=tensor_transform)

    x_train, y_train = zip(*train_set)
    x_train, y_train = torch.cat(x_train), torch.Tensor(y_train)
    x_test, y_test = zip(*test_set)
    x_test, y_test = torch.cat(x_test), torch.Tensor(y_test)

    return x_train, y_train, x_test, y_test

def load_twomoon() -> tuple:
        data, y = make_moons(n_samples=7000, shuffle=True, noise=0.075, random_state=None)
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.33, random_state=42)
        x_train, x_test = torch.Tensor(x_train), torch.Tensor(x_test)
        y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)
        return x_train, y_train, x_test, y_test



def load_reuters() -> tuple:
    with h5py.File('../data/Reuters/reutersidf_total.h5', 'r') as f:
        x = np.asarray(f.get('data'), dtype='float32')
        y = np.asarray(f.get('labels'), dtype='float32')

        n_train = int(0.9 * len(x))
        x_train, x_test = x[:n_train], x[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

    x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
    y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test

def read_data(path):
    if path.endswith('npy'):
        data = np.load(path)
    elif path.endswith('off'):
        data = read_off(path)
    else:
        data = np.loadtxt(path, delimiter=',', dtype=np.float32)
    return data
def load_from_path(dpath: str, lpath: str = None, test_eq_train=False) -> tuple:
    X = read_data(dpath)

    if test_eq_train and lpath is not None:
        y = np.load(lpath) if lpath.endswith('npy') else np.loadtxt(lpath, delimiter=',', dtype=np.float32)
        x_train, x_test = X.astype(np.float32), X.astype(np.float32)
        x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
        y_train, y_test = y, y
        y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)
        return x_train, y_train, x_test, y_test

    if lpath is not None:
        y = np.load(lpath) if lpath.endswith('npy') else np.loadtxt(lpath, delimiter=',', dtype=np.float32)
        n_train = max(int(0.9 * len(X)), y.shape[1])
        x_train, x_test = X[:n_train].astype(np.float32), X[n_train:].astype(np.float32)
        x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
        y_train, y_test = y[:n_train], y[n_train:]
        y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)

    else:
        n_train = int(0.9 * len(X))
        x_train, x_test = X[:n_train].astype(np.float32), X[n_train:].astype(np.float32)
        x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
        y_train, y_test = None, None


    # x_train, x_test = mean_std_normalize_data(x_train, x_test, mean=0.5)
    # x_train, x_test = minmax_normalize_data(x_train, x_test)
    # x_train, x_test = normalize_data(x_train, x_test)

    return x_train, y_train, x_test, y_test


def normalize_data(x_train, x_test):
    x_train_mean = torch.mean(x_train, dim=0)
    x_train_std = torch.std(x_train, dim=0)

    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    return x_train, x_test

def minmax_normalize_data(x_train, x_test):
    x_train_min = torch.min(x_train, dim=0).values
    x_train_max = torch.max(x_train, dim=0).values

    x_train = (x_train - x_train_min) / (x_train_max - x_train_min)
    x_test = (x_test - x_train_min) / (x_train_max - x_train_min)

    return x_train, x_test

def mean_std_normalize_data(x_train, x_test, mean=0.5, std=0.5):
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    return x_train, x_test

def load_data(dataset: str, test_eq_train=False) -> tuple:
    """
    This function loads the dataset specified in the config file.
    

    Args:
        dataset (str or dictionary):    In case you want to load your own dataset, 
                                        you should specify the path to the data (and label if applicable) 
                                        files in the config file in a dictionary fashion under the key "dataset".

    Raises:
        ValueError: If the dataset is not found in the config file.

    Returns:
        tuple: A tuple containing the train and test data and labels.
    """

    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = load_mnist()
    elif dataset == 'twomoons':
        x_train, y_train, x_test, y_test = load_twomoon()
    elif dataset == 'reuters':
        x_train, y_train, x_test, y_test = load_reuters()
    else:
        try:
            data_path = dataset["dpath"]
            if "lpath" in dataset:
                label_path = dataset["lpath"]
            else:
                label_path = None
        except:
            raise ValueError("Could not find dataset path. Check your config file.")
        x_train, y_train, x_test, y_test = load_from_path(data_path, label_path, test_eq_train)

    return x_train, x_test, y_train, y_test

def read_off(filepath):
    """
    read a standard .off file

    Parameters
    -------------------------
    file : path to a '.off'-format file

    Output
    -------------------------
    vertices,faces : (n,3), (m,3) array of vertices coordinates
                    and indices for triangular faces
    """
    with open(filepath, 'r') as f:
        if f.readline().strip() != 'OFF':
            raise TypeError('Not a valid OFF header')
        n_verts, _, _ = [int(x) for x in f.readline().strip().split(' ')]
        vertices = [[float(x) for x in f.readline().strip().split()] for _ in range(n_verts)]
        # if n_faces > 0:
        #     faces = [[int(x) for x in f.readline().strip().split()][1:4] for _ in range(n_faces)]
        #     faces = np.asarray(faces)
        # else:
        #     faces = None

    return np.asarray(vertices) #, faces