import numpy as np
from skimage.feature import peak_local_max
import cv2
import torch
from models.Unet_model import UNet
import os.path as osp

def non_max_suppression(image, size_filter, proba):
    non_max = peak_local_max(image, min_distance=size_filter, threshold_abs=proba, \
                      exclude_border=True, indices=False)
    kp = np.where(non_max>0)
    if len(kp[0]) != 0:
        for i in range(len(kp[0]) ):

            window=non_max[kp[0][i]-size_filter:kp[0][i]+(size_filter+1), \
                           kp[1][i]-size_filter:kp[1][i]+(size_filter+1)]
            if np.sum(window)>1:
                window[:,:]=0
    return non_max


def sift(image, kp_before):
    # this is root SIFT
    SIFT = cv2.xfeatures2d.SIFT_create()
    kp, des = SIFT.compute(image, kp_before)
    if des is not None:
        eps = 1e-7
        des /= (des.sum(axis=1, keepdims=True) + eps)
        des = np.sqrt(des)
    return kp, des


class GLAMpointsInference:
    def __init__(self, path_weights, nms, min_prob):

        self.nms = nms
        self.min_prob = min_prob
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        net = UNet()
        # load the pretrained weights to the network
        self.path_weights = path_weights
        if not osp.isfile(self.path_weights):
            raise ValueError('check the snapshots path, checkpoint is {}'.format(self.path_weights))
        try:
            net.load_state_dict(torch.load(self.path_weights))
        except:
            net.load_state_dict(torch.load(self.path_weights)['state_dict'])
        print('successfully loaded weights to the network !')

        net.eval()
        self.net = net.to(self.device)

    def pre_process_data(self, image):

        if torch.is_tensor(image):
            # if it is already a Torch Tensor
            if len(image.shape) == 2:
                # gray image
                image_norm = image.float() / float(image.max())
                image_norm = image_norm.unsqueeze(0).unsqueeze(0).float().to(self.device)
            if len(image.shape) == 3:
                # BxHxW, HxWx3 or 3xHxW
                if image.shape[0] == 3:
                    # 3xHxW
                    image_numpy = image.permute(1,2,0).cpu().numpy() # now HxWx3
                    image = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2GRAY)
                    image_norm = np.float32(image) / float(np.max(image))
                    # creates a Torch input tensor of dimension 1x1xHxW
                    image_norm = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0).float().to(self.device)
                elif image.shape[2] == 3:
                    # HxWx3
                    image_numpy = image.cpu().numpy()
                    image = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2GRAY)
                    image_norm = np.float32(image) / float(np.max(image))
                    # creates a Torch input tensor of dimension 1x1xHxW
                    image_norm = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0).float().to(self.device)
                else:
                    # BxHxW, already gray scale
                    image_norm = image.float() / float(image.max())
                    image_norm = image_norm.unsqueeze(1).float().to(self.device)
            if len(image.shape) == 4:
                # Bx1xHxW or BxHxWx1 or Bx3xHxW or BxHxWx3
                if image.shape[1] == 1:
                    # already good Bx1xHxW
                    image_norm = image.float() / float(image.max())
                elif image.shape[3] == 1:
                    # BxHxWx1
                    image_norm = image.permute(0,3,1,2).float() / float(image.max())
                else:
                    if image.shape[1] == 3:
                        # Bx3xHxW
                        image = image.permute(0, 2, 3, 1)
                    # now tensor BxHxWx3
                    B,H,W,_ = image.shape
                    image_gray = torch.Tensor((), dtype=torch.float32)
                    image_gray.new_zeros((B,H,W))
                    for i in image.shape[0]:
                        image_numpy_i = image[i].cpu().numpy()
                        image_i = cv2.cvtColor(image_numpy_i, cv2.COLOR_RGB2GRAY)
                        image_norm_i = np.float32(image_i) / float(np.max(image))
                        image_gray[i] = image_norm_i

                    image_norm = image_gray.unsqueeze(1).float().to(self.device)

        elif isinstance(image, type(np.empty(0))):
            if len(image.shape) != 2:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image_norm = np.float32(image) / float(np.max(image))

            # creates a Torch input tensor of dimension 1x1xHxW
            image_norm = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0).float().to(self.device)
        return image_norm.float().to(self.device)

    def find_and_describe_keypoints(self, image):
        image_preprocessed = self.pre_process_data(image)
        kp_map = self.net(image_preprocessed).to(self.device)
        # output is of shape 1x1xHxW

        kp_map_nonmax = non_max_suppression(kp_map.data.cpu().numpy().squeeze(), self.nms, self.min_prob)

        keypoints_map = np.where(kp_map_nonmax > 0)
        kp_array = np.array([keypoints_map[1], keypoints_map[0]]).T
        kp_cv2 = [cv2.KeyPoint(kp_array[i, 0], kp_array[i, 1], 10) for i in range(len(kp_array))]
        kp, des = sift(np.uint8(image), kp_cv2)
        kp = np.array([m.pt for m in kp], dtype=np.int32)
        return kp, des


class SIFT_noorientation:
    def __init__(self, **kwargs):
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=int(kwargs['nfeatures']),
                                                contrastThreshold=float(kwargs['contrastThreshold']),
                                                edgeThreshold= float(kwargs['edgeThreshold']),
                                                sigma=float(kwargs['sigma']))

    def find_and_describe_keypoints(self, image):
        eps = 1e-7
        kp = self.sift.detect(image, None)
        kp = np.int32([kp[i].pt for i in range(len(kp))])
        kp = [cv2.KeyPoint(kp[i, 0], kp[i, 1], 10) for i in range(len(kp))]
        kp, des = self.sift.compute(image, kp)
        if des is not None:
            des /= (des.sum(axis=1, keepdims=True) + eps)
            des = np.sqrt(des)
        kp = np.array([kp[i].pt for i in range(len(kp))])
        return kp, des


class SIFT:
    def __init__(self, **kwargs):
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=int(kwargs['nfeatures']),
                                                contrastThreshold=float(kwargs['contrastThreshold']),
                                                edgeThreshold= float(kwargs['edgeThreshold']),
                                                sigma=float(kwargs['sigma']))

    def find_and_describe_keypoints(self, image):
        eps = 1e-7
        kp = self.sift.detect(image, None)
        kp, des = self.sift.compute(image, kp)
        if des is not None:
            des /= (des.sum(axis=1, keepdims=True) + eps)
            des = np.sqrt(des)
        kp = np.array([kp[i].pt for i in range(len(kp))])
        return kp, des