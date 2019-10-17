from __future__ import print_function, division
import numpy as np
import os
import cv2
import dlib


def find_face_landmarks(input_path, predictor_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    img = dlib.load_rgb_image(input_path)
    dets = detector(img, 1)
    try:
        shape = predictor(img, dets[0])
    except IndexError:
        print("No face landmarks were found.")
        return []
    return list(map(lambda p: np.matrix([p.x, p.y]), shape.parts()))


def get_three_face_coordinates_from_68_face_landmarks(landmarks):
    """
    The key points for face alignment we used are the two for the center of the eyes and
    the average point of the corners of the mouth
    :param landmarks: list of numpy matrices returned by find_face_landmarks function
    :return: list of three lists with required coordinates
    """
    left_eye = np.mean(landmarks[36:42], axis=0)[0]
    right_eye = np.mean(landmarks[42:48], axis=0)[0]
    mouth = np.mean(landmarks[48:], axis=0)[0]
    return [left_eye, right_eye, mouth]


def transformation_from_points(points1, scale):
    points = [[70, 112],
              [110, 112],
              [90,  150]]
    points2 = np.array(points) * scale
    points2 = points2.astype(np.float64)
    points1 = points1.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(np.matmul(points1.T, points2))
    R = (np.matmul(U, Vt)).T
    sR = (s2 / s1) * R
    T = c2.reshape(2, 1) - (s2 / s1) * np.matmul(R, c1.reshape(2, 1))
    M = np.concatenate((sR, T), axis=1)
    return M


class ImageLoader(object):
    def __init__(self, mode='test'):
        self.scale = 2
        self.crop_height = 134 * self.scale
        self.crop_width = 134 * self.scale
        self.crop_center_y_offset = 10 * self.scale
        self.output_scale = (260, 260)
        self.ori_scale = (178 * self.scale, 218 * self.scale)
        self.random_x = 0
        self.random_y = 0
        self.flip = 0
        if mode == 'train':
            self.flip = np.random.randint(0, 2)
            self.random_x = np.random.randint(-3, 4)
            self.random_y = np.random.randint(-3, 4)

    def image_loader(self, path, points):
        if os.path.exists(path):
            img = cv2.imread(path)
            three_points = np.zeros((3, 2))
            three_points[0] = points[0]  # the location of the left eye
            three_points[1] = points[1]  # the location of the right eye
            three_points[2] = points[2]  # the location of the center of the mouth
            three_points.astype(np.float32)
            M = transformation_from_points(three_points, self.scale)
            align_img = cv2.warpAffine(img, M, self.ori_scale, borderValue=[127, 127, 127])
            l = int(round(self.ori_scale[0] / 2 - self.crop_width / 2 + self.random_x))
            r = int(round(self.ori_scale[0] / 2 + self.crop_width / 2 + self.random_x))
            t = int(round(self.ori_scale[1] / 2 - self.crop_height / 2 + self.crop_center_y_offset + self.random_y))
            d = int(round(self.ori_scale[1] / 2 + self.crop_height / 2 + self.crop_center_y_offset + self.random_y))
            align_img2 = align_img[t:d, l:r, :]
            align_img2 = cv2.resize(align_img2, self.output_scale)
            return align_img2
        else:
            raise Exception("image = 0")


if __name__ == '__main__':
    image_path = '../face.bmp'
    landmarks = find_face_landmarks(image_path, 'shape_predictor_68_face_landmarks.dat')
    points = get_three_face_coordinates_from_68_face_landmarks(landmarks)
    im_loader = ImageLoader(mode='test')
    img = im_loader.image_loader(image_path, points)
    cv2.imshow('', img)
    cv2.waitKey()
