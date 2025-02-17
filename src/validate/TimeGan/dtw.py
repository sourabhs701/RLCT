from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def compute_dtw(real_data, synthetic_data):
    distance, path = fastdtw(real_data, synthetic_data, dist=euclidean)
    return distance
