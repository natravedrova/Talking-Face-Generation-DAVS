import dlib


def align_face(input_path, predictor_path='shape_predictor_68_face_landmarks.dat'):
    """
    TODO: there is dlib.save_face_chip function that aligns and save an image, use it without additional save function
    Align a face in the input photo
    :param input_path: path to image with a face
    :param predictor_path: path to trained facial shape predictor
    :return: RGB image of aligned face or None if a face was not found
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    img = dlib.load_rgb_image(input_path)
    dets = detector(img, 1)
    if 1 > len(dets):
        print("Sorry, there were no faces found in '{}'".format(input_path))
        return None
    image = dlib.get_face_chip(img, predictor(img, dets[0]), size=256, padding=0.25)
    return image


def align_and_save_face(input_path, output_path):
    """
    Align face and save aligned version to the output_path
    :param input_path: path to image with a face
    :param output_path: output path
    :return: None
    """
    aligned_image = align_face(input_path)
    if aligned_image is not None:
        dlib.save_image(aligned_image, output_path)
    return


if __name__ == '__main__':
    align_and_save_face('../face.bmp', '../face_aligned.bmp')


