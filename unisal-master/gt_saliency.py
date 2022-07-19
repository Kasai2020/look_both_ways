#gt saliency


import numpy as np
import cv2
import open3d
from os import listdir
from os.path import isfile, join

def get_eye_and_dir(label_path):
    with open(label_path, 'r') as infile:
        label_gaze = [[x.replace("[", "").replace("]", "").replace(":", "").replace(",", "") for x in line.split()] for line in infile]
        label_gaze = [list(filter(None, label_gaze_x)) for label_gaze_x in label_gaze]
        if len(label_gaze) == 5:
            g_2d = np.asarray(label_gaze[0][1:], dtype=np.float64, order='C')
            label = np.asarray(label_gaze[2][1:], dtype=np.float64, order='C')
            eye_loc = np.asarray(label_gaze[4][1:], dtype=np.float64, order='C')

        else:
            #label_1 = np.array(map(float, label_gaze[2][1:]))
            g_2d = np.asarray(label_gaze[0][1:], dtype=np.float64, order='C')
            label_1 = np.asarray(label_gaze[2][1:], dtype=np.float64, order='C')
            label_2 = np.asarray(label_gaze[5][1:], dtype=np.float64, order='C')
            label = (label_1 + label_2) / 2

            eye_loc_1 = np.asarray(label_gaze[4][1:], dtype=np.float64, order='C')
            eye_loc_2 = np.asarray(label_gaze[7][1:], dtype=np.float64, order='C')
            eye_loc = (eye_loc_1 + eye_loc_2) / 2

    return eye_loc, label, g_2d

def get_saliency_from_gaze(e, g, R, K_inv, depth_rect, k):
    # Get 3D points from depth map
    #X = d(x)K_inv @ x
    X,Y = np.mgrid[0:942, 0:489]
    xy = np.vstack((X.flatten(order='C'), Y.flatten(order='C'))).T
    z = np.reshape(np.ones(489*942), ((489*942), 1))
    xyz = np.hstack((xy, z))


    xyz_3D_flat = np.dot(K_inv,xyz.T).T

    xyz_3D = np.reshape(xyz_3D_flat, (942,489,3), order='C')
    xyz_3D = np.transpose(xyz_3D,(1,0,2))

    depth_rect_mult = np.reshape(depth_rect, (489,942,1))
    xyz_3D = np.transpose(xyz_3D, (2, 0 , 1))
    depth_rect_mult = np.transpose(depth_rect_mult, (2, 0 , 1))


    xyz_3D = np.multiply(depth_rect_mult, xyz_3D)
    xyz_3D = np.transpose(xyz_3D, (1, 2, 0))
    X = xyz_3D


    # #X_3D = np.reshape(X, ((489*942), 3))
    # #
    # # pcd_cam_in = open3d.geometry.PointCloud()
    # # pcd_cam_in.points = open3d.utility.Vector3dVector(X_3D)
    # # open3d.io.write_point_cloud("pcd_cam_in.ply", pcd_cam_in)
    #
    # print(np.shape(depth_rect))
    # X_rec = np.zeros((np.shape(depth_rect)[0], np.shape(depth_rect)[1], 3))
    # for u in range (0,len(depth_rect[0])):
    #     for v in range(0,len(depth_rect)):
    #         depth_value = depth_rect[v][u]
    #         X_rec[v,u] = (depth_value * (K_inv @ np.asarray([[u],[v],[1]]))).ravel()
    #         #print(((K_inv @ np.asarray([[u],[v],[1]]))).ravel())
    #         #X_rec[v,u] = ((K_inv @ np.asarray([[u],[v],[1]]))).ravel()
    #
    # #X_rec_test = depth_rect[v][u] *
    # # X_rec = np.reshape(X_rec,(np.shape(depth_rect)[0] * np.shape(depth_rect)[1], 3) )
    # # X_rec = (r_real @ X_rec.T).T
    # # X_rec = X_rec.T
    # # X_unrect = np.append(X_rec, np.ones((1, X_rec.shape[1])), axis=0)
    # # X_unrect_w = transform_out @ X_unrect
    # # X_unrect_w = X_unrect_w[0:3].transpose()
    # # pcd_out_test = open3d.geometry.PointCloud()
    # # pcd_out_test.points = open3d.utility.Vector3dVector(X_unrect_w)
    # # open3d.io.write_point_cloud("pcd_out.ply", pcd_out_test)
    # X = X_rec


    # Get 3D s vector from each 3D point
    #s(x) = R (X-e)/||X-e||

    X = np.reshape(X, ((489*942), 3))
    X = X - e
    X_norm = np.linalg.norm(X, axis=1)

    X = np.divide(X.T, np.reshape(X_norm,(1,(489*942))))
    X = X.T
    X = (R @ X.T).T

    s_X = X

    # Get saliency value from 3D s vectors and gaze direction g
    s_g = np.exp(k * (np.transpose(g) @ np.transpose(s_X)))
    sum = np.sum(s_g)
    s_g = s_g / sum
    s_g = s_g.reshape((489,942))


    return s_g





    # ok
    # depth_value = true_depth[int(np.around(xy_azure[1])), int(np.around(xy_azure[0]))]
    # gaze_point_true = (depth_value * (K_inv @ np.asarray([[int(np.around(xy_azure[0]))], [int(np.around(xy_azure[1]))],[1]])))
    # gaze_point_true = (r_real @ gaze_point_true)
    # gaze_point_true = gaze_point_true.flatten()
    # gaze_point_true_unrect = np.append(gaze_point_true, [1], axis=0)
    # gaze_point_true_unrect_w = transform_out @ gaze_point_true_unrect
    # gaze_point_true_unrect_w = gaze_point_true_unrect_w[0:3].transpose()
    # pcd_gaze_true = open3d.geometry.PointCloud()
    # pcd_gaze_true.points = open3d.utility.Vector3dVector([gaze_point_true_unrect_w])


if __name__ == "__main__":
    H1 = np.load('calib/H1.npy')
    H2 = np.load('calib/H2.npy')
    k1 = np.load('calib/k.npy')
    k2 = np.load('calib/k2.npy')
    r_real = np.load('calib/r.npy')
    t_real = np.load('calib/t.npy')
    b = np.load('calib/b.npy')
    transform_in = np.load("calib/transform_in.npy")
    transform_out = np.load("calib/transform_out.npy")
    K = np.load("calib/k.npy")


    #out = cv2.VideoWriter('sal.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (1884, 489))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('sal.avi', fourcc, 5, (1884, 489))


    #BenJ 3 4204
    #Gaze_Loc_2D: [273, 334]
    Right_Gaze_Dir = [ 0.11040104, -0.1401288,  -0.98395911]
    Right_3D_Eye_Loc = [-0.00490919,  0.00402753,  0.65675006]
    # Left_Gaze_Dir = [ 0.10728173, -0.1404216,  -0.98426237]
    # Left_3D_Eye_Loc = [0.06076135, 0.00816689, 0.64975006]
    #print(Right_Gaze_Dir)
    onlyfiles = [f for f in listdir("/media/isaac/Extreme SSD/Final_Data/train/Benjamin_Jourdan_11-14-2021_3_data/gaze_info") if isfile(join("/media/isaac/Extreme SSD/Final_Data/train/Benjamin_Jourdan_11-14-2021_3_data/gaze_info", f))]

    #print(onlyfiles)
    for num in onlyfiles:
        number = num[:-9]

        e, g, g_2d = get_eye_and_dir("/media/isaac/Extreme SSD/Final_Data/train/Benjamin_Jourdan_11-14-2021_3_data/gaze_info/" + number + "_gaze.txt")

        #4264
        depth_rect = np.load("/media/isaac/Extreme SSD/Final_Data/train/Benjamin_Jourdan_11-14-2021_3_data/scene_depth/" + number + "_depth.npy")
        scene_rect = cv2.imread("/media/isaac/Extreme SSD/Final_Data/train/Benjamin_Jourdan_11-14-2021_3_data/scene_ims/" + number + "_scene.png")
        #
        # e = np.asarray(Right_3D_Eye_Loc)
        # g = np.asarray(Right_Gaze_Dir)


        i_T_o = np.linalg.inv(transform_in) @ transform_out
        R = i_T_o[:3,:3]
        R = R @ r_real
        K_inv = np.linalg.inv(K)

        k = 20
        s_g = get_saliency_from_gaze(e, g, R, K_inv, depth_rect, k)

        s_g_vis = (s_g / np.max(s_g)) * 255

        #cv2.circle(s_g_vis, ( 374,409), 5, (0,0,255), 1)
        s_g_vis = np.float32(s_g_vis)
        s_g_vis = cv2.cvtColor(s_g_vis,cv2.COLOR_GRAY2RGB)
        cv2.circle(scene_rect, g_2d.astype(int), 5, (0,0,255), 3)
        combined = np.concatenate((scene_rect, s_g_vis), axis=1)
        #cv2.imwrite("sal_ims/" + number + ".png", combined)
        combined = np.uint8(combined)
        out.write(combined)
    out.release()
