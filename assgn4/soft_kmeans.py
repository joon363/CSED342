import os
import math
from utils import converged, plot_2d_soft, plot_centroids, read_data, \
    load_centroids, write_centroids_tofile
import matplotlib.pyplot as plt
import numpy as np

from kmeans import euclidean_distance

# problem for students
def get_responsibility(data_point, centroids, beta):
    """Calculate the responsibiliy of each cluster for a single data point.
    You should use the euclidean_distance function (that you previously implemented).
    You can use the math.exp() function to calculate the responsibility.

    Arguments:
        data_point: a list of floats representing a data point
        centroids: a dictionary representing the centroids where the keys are
                   strings (centroid names) and the values are lists of
                   centroid locations
        beta: hyper-parameter

    Returns: a dictionary whose keys are the the centroids' key names and
             value is a float as the responsibility of the cluster for the data point.
    """
    # initializing 
    expdist=0
    expdist_sum=0
    soft_dict={}
    # calculate exp(-beta*dist) for each centroid
    for name,loc in centroids.items():
        expdist=math.exp((-1)*beta*euclidean_distance(data_point,loc))
        expdist_sum = expdist_sum + expdist
        soft_dict[name]=expdist
    # divide by sum; responsibility
    for name,val in soft_dict.items():
        soft_dict[name]=(val/expdist_sum)

    return soft_dict


# problem for students
def update_soft_assignment(data, centroids, beta):
    """Find the responsibility of each cluster for all data points.
    You should use the get_responsibility function (that you previously implemented).

    Arguments:
        data: a list of lists representing all data points
        centroids: a dictionary representing the centroids where the keys are
                   strings (centroid names) and the values are lists of
                   centroid locations

    Returns: a dictionary whose keys are the data points of type 'tuple'
             and values are the dictionary returned by get_responsibility function.
             (In python, 'list' cannot be the 'key' of 'dict')
             
    """
    # dictionary
    all_dict = {}
    for d in data:
        # add (point):{centroid:resp}'s  into dictionary
        key = tuple(d)
        all_dict[key]=get_responsibility(d,centroids,beta)
    return all_dict

    
            

# problem for students
def update_centroids(soft_assignment_dict):
    """Update centroid locations with the responsibility of the cluster for each point
    as a weight. You can numpy methods for simple array computations. But the values of 
    the result dictionary must be of type 'list'.

    Arguments:
        assignment_dict: the dictionary returned by update_soft_assignment function

    Returns: A new dictionary representing the updated centroids
    """
    new_dict= {}
    dim=0
    # convert 
    # {(x1, y1): {'c1': 0.8, 'c2': 0.7},
    #  (x2, y2): {'c1': 0.5, 'c2': 0.6}}
    # into
    # {c1: {(x1, y1): 0.8, (x2, y2): 0.5}, 
    #  c2: {(x1, y1): 0.7, (x2, y2): 0.6}}

    for point, centroid_dict in soft_assignment_dict.items():
        dim=len(point)
        for centroid, resp in centroid_dict.items():
            if centroid not in new_dict:
                new_dict[centroid] = {}
            new_dict[centroid][point] = resp


    # and then calculate per centroid
    final_dict = {}
    for centroid, values in new_dict.items():
        resp_sum = 0
        weight_point = [0]*dim
        # weighted points sum
        for point, resp in values.items():
            weight_point=[x1+resp*x2 for x1, x2 in zip(weight_point,point)]
            resp_sum = resp_sum+resp
        # divide by sum of resp
        weight_point=[x/resp_sum for x in weight_point]
        final_dict[centroid] = weight_point
        
    return final_dict

def main(data, init_centroids):
    #######################################################
    # You do not need to change anything in this function #
    #######################################################
    beta = 50
    centroids = init_centroids
    old_centroids = None
    total_step = 7
    for step in range(total_step):
        # save old centroid
        old_centroids = centroids
        # new assignment
        soft_assignment_dict = update_soft_assignment(data, old_centroids, beta)
        # update centroids
        centroids = update_centroids(soft_assignment_dict)
        # plot centroid
        fig = plot_2d_soft(soft_assignment_dict, centroids)
        plt.title(f"step{step}")
        fig.savefig(os.path.join("results", "2D_soft", f"step{step}.png"))
        plt.clf()
    print(f"{total_step} iterations were completed.")
    return centroids


if __name__ == '__main__':
    data, label = read_data("data/data_2d.csv")
    init_c = load_centroids("data/2d_init_centroids.csv")
    final_c = main(data, init_c)
    write_centroids_tofile("2d_final_centroids_with_soft_kmeans.csv", final_c)
