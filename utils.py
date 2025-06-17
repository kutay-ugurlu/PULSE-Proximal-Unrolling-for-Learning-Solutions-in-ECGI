import numpy as np
from numpy.linalg import inv as inv

def add_noise(data, noise_param, flag: str = "SNR"):

    assert len(data.shape) <= 2, "The input dimension should be less than 3."
    np.random.seed(101)

    match flag:

        case "SNR":
            SNR = noise_param
            avg_pow_data = np.linalg.norm(data[:])/np.sqrt(data.size)
            std_noise = avg_pow_data / 10**(SNR/20)

        case "STD_N":
            std_noise = noise_param

        case _:
            raise ValueError("Invalid noise parameter")

    noise = np.random.randn(*data.shape)

    # Obtain 0 mean noise with covariance std_noise**2*I
    noise = (noise-np.mean(noise, axis=1, keepdims=True)) / \
        (np.linalg.norm(noise[:])/np.sqrt(noise.size)) * std_noise

    return data+noise, noise, np.std(noise)

def create_moments(data:np.ndarray):
    
    # Assumes data of size Nodes by Timeframes 
    n,t = data.shape
    mean = np.mean(data, axis=1, keepdims=True)
    zero_mean_data = data - mean
    covariance = zero_mean_data.dot(zero_mean_data.T) / (t-1)
    
    return covariance, mean
    
def map_solution(A,Cx,Cn,y,meanX):
    
    B = A.T.dot(inv(Cn))
    return inv(B.dot(A) + inv(Cx)).dot(B.dot(y) + inv(Cx).dot(meanX))


