import sys
import json
import torch
import random
import numpy as np

from utils import *
from data import load_data
from metrics import Metrics
from sklearn.cluster import KMeans
from SpectralNet import SpectralNet
from scipy.spatial.distance import cdist
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_FILENAME = 'spectralnet.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG, format=LOG_FORMAT)


class InvalidMatrixException(Exception):
    pass


def set_seed(seed: int = 0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    config_path = sys.argv[1]
    with open (config_path, 'r') as f:
        config = json.load(f)

    dataset = config["dataset"]
    n_clusters = config["n_clusters"]
    should_check_generalization = config["should_check_generalization"]
    
    x_train, x_test, y_train, y_test = load_data(dataset)

    if not should_check_generalization:
        if y_train is None:
            x_train = torch.cat([x_train, x_test])
            
        else:
            x_train = torch.cat([x_train, x_test])
            y_train = torch.cat([y_train, y_test])
    
    try:
        spectralnet = SpectralNet(n_clusters=n_clusters, config=config)
        spectralnet.fit(x_train, y_train)

    except torch._C._LinAlgError:
        raise InvalidMatrixException("The output of the network is not a valid matrix to the orthogonalization layer. " 
                                     "Try to decrease the learning rate to fix the problem.") 

    if not should_check_generalization:

        cluster_assignments = spectralnet.predict(x_train)
        plot_data_by_assignmets(x_train, cluster_assignments)

        if y_train is not None:
            y = y_train.detach().cpu().numpy()
            acc_score = Metrics.acc_score(cluster_assignments, y, n_clusters)
            nmi_score = Metrics.nmi_score(cluster_assignments, y)
            print(f"ACC: {np.round(acc_score, 3)}")
            print(f"NMI: {np.round(nmi_score, 3)}")

        return spectralnet.embeddings_, cluster_assignments

    else:
        y_test = y_test.detach().cpu().numpy()
        spectralnet.predict(x_train)
        train_embeddings = spectralnet.embeddings_
        test_assignments = spectralnet.predict(x_test)
        test_embeddings = spectralnet.embeddings_
        kmeans_train = KMeans(n_clusters=n_clusters).fit(train_embeddings)
        dist_matrix = cdist(test_embeddings, kmeans_train.cluster_centers_)
        closest_cluster = np.argmin(dist_matrix, axis=1)
        acc_score = Metrics.acc_score(closest_cluster, y_test, n_clusters)
        nmi_score = Metrics.nmi_score(closest_cluster, y_test)
        print(f"ACC: {np.round(acc_score, 3)}")
        print(f"NMI: {np.round(nmi_score, 3)}")

        return test_embeddings, test_assignments


if __name__ == "__main__":
    embeddings, assignments = main()
    write_assignmets_to_file(assignments)
    np.save('emb_lion1.npy', embeddings)
    print("Your assignments were saved to the file 'cluster_assignments.csv!")
    # create_mp4_from_directory(
    # r"C:\Users\yanir\PycharmProjects\SpectralNet\vectors", r'C:\Users\yanir\PycharmProjects\SpectralNet\movie.mp4'
# )