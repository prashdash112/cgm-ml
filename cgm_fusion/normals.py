
# from pyntcloud import PyntCloud




# examples/Python/Basic/pointcloud.py

import numpy as np
import open3d as o3d
from timeit import default_timer as timer

from pyntcloud import PyntCloud


if __name__ == "__main__":
    mat = np.array([[11,12,13],[21,22,23],[31,32,33]])
    print("Original matrix")
    print(mat)
    mat_slice = np.array(mat[:2,:2]) # Notice the np.array method

    pc_path = "/data/home/cpfitzner/test.pcd"

    #  load the file with pyntcloud
    cloud_pyntcloud  = PyntCloud.from_file(pc_path) 
    
    np_array = cloud_pyntcloud.points
    
    print(type(np_array))

    np_array_cropped = np_array[:2, :2]
    print(np_array_cropped.shape)

    # print("Load a ply point cloud, print it, and render it")
    # #pcd = o3d.io.read_point_cloud("/data/home/cpfitzner/test.pcd")
    # pcd = o3d.geometry.PointCloud()
    # print(pcd)
    # print(np.asarray(pcd.points))
    # pcd.points = o3d.utility.Vector3dVector(np_array)
    # #o3d.visualization.draw_geometries([pcd])

    # print("Downsample the point cloud with a voxel of 0.05")
    # downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    # downpcd = pcd
    
    # #o3d.visualization.draw_geometries([downpcd])
    # start = timer()


    # print("Recompute the normal of the downsampled point cloud")
    # downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # end = timer()
    # print(end - start)
    # print("Print a normal vector of the 0th point")
    # print(downpcd.normals[0])
    # print("Print the normal vectors of the first 10 points")
    # print(np.asarray(downpcd.normals)[:10, :])
    # print("")



# ''' 
#     calculate the normals of a given point cloud. The implementation uses pyntcloud as framework.
#     This method searches for the 
# '''
# def get_normals(cloud, nr_of_neighbors=4):
    




#     # extract the normals ; the idea is based on the following code
#     # https://github.com/daavoo/pyntcloud/issues/5
#     k_neighbors = cloud.get_neighbors(k=nr_of_neighbors)                        # get 10 nearest neighbors
#     print ("neighbors:")
#     print (k_neighbors)
#     cloud.add_scalar_field("normals", k_neighbors=k_neighbors)                  # add a scala field to the normals




#     return normals







# ''' 
#     main for testing
#     has to be removed afterwards
# '''
# def main():

#     # load point cloud for testing
#     cloud = PyntCloud.from_file("/data/home/cpfitzner/test.pcd")

#     print (cloud)

#     # append normals to point cloud
#     get_normals(cloud, nr_of_neighbors=10)



#     print('finished ending')




# if __name__ == "__main__":
#     main()

