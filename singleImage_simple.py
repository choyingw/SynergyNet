import cv2
from synergy3DMM import SynergyNet
    

if __name__ == '__main__':
    model = SynergyNet()
    I=cv2.imread('img/sample_2.jpg', -1)
    # get landmark [[y, x, z], 68 (points)], mesh [[y, x, z], 53215 (points)], and face pose (Euler angles [yaw, pitch, roll] and translation [y, x, z])
    lmk3d, mesh, pose = model.get_all_outputs(I)
    print(lmk3d[0].shape)
    print(mesh[0].shape)
    print(pose[0])
