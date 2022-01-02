import glob
import argparse
from pathlib import Path

import dlib

import numpy as np
import PIL.Image
import scipy.ndimage

import pdb

"""
    pip install dlib
    # download face landmark model from:
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""


def get_landmark(filepath, predictor):
    """
    get landmark with dlib
    return: np.array shape=(68, 2)
    """

    detector = dlib.get_frontal_face_detector()

    image = dlib.load_rgb_image(filepath)
    # image.shape -- (250, 250, 3)

    dets = detector(image, 1)
    # dets -- rectangles[[(67, 80) (175, 187)]]

    filepath = Path(filepath)
    print(f"{filepath.name}: {len(dets)} face detected")

    shapes = [predictor(image, d) for k, d in enumerate(dets)]
    # type(shapes), len(shapes), shapes[0].rect
    # (<class 'list'>, 1, (67,80,175,187)

    lms = [np.array([[tt.x, tt.y] for tt in shape.parts()]) for shape in shapes]
    # len(lms), type(lms[0]), lms[0].shape
    # (1, <class 'numpy.ndarray'>, (68, 2))

    return lms


def align_face(filepath, predictor):
    """
    param filepath: str
    return: list of PIL Images
    """

    lms = get_landmark(filepath, predictor)
    imgs = []
    for lm in lms:
        lm_chin = lm[0:17]  # left-right
        lm_eyebrow_left = lm[17:22]  # left-right
        lm_eyebrow_right = lm[22:27]  # left-right
        lm_nose = lm[27:31]  # top-down
        lm_nostrils = lm[31:36]  # top-down

        lm_eye_left = lm[36:42]  # left-clockwise
        lm_eye_right = lm[42:48]  # left-clockwise
        lm_mouth_outer = lm[48:60]  # left-clockwise
        lm_mouth_inner = lm[60:68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # (Pdb) quad
        # array([[ 34.55553295,  36.46790782],
        #        [ 35.19290782, 203.61946705],
        #        [202.34446705, 202.98209218],
        #        [201.70709218,  35.83053295]])
        # (Pdb) qsize
        # 167.15277443105754

        # read image
        image = PIL.Image.open(filepath)

        # output_size = 1024
        # transform_size = 4096
        output_size = int(qsize)
        transform_size = 4 * output_size
        enable_padding = True

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (
                int(np.rint(float(image.size[0]) / shrink)),
                int(np.rint(float(image.size[1]) / shrink)),
            )
            image = image.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (
            int(np.floor(min(quad[:, 0]))),
            int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))),
        )
        crop = (
            max(crop[0] - border, 0),
            max(crop[1] - border, 0),
            min(crop[2] + border, image.size[0]),
            min(crop[3] + border, image.size[1]),
        )
        if crop[2] - crop[0] < image.size[0] or crop[3] - crop[1] < image.size[1]:
            image = image.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (
            int(np.floor(min(quad[:, 0]))),
            int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))),
        )
        pad = (
            max(-pad[0] + border, 0),
            max(-pad[1] + border, 0),
            max(pad[2] - image.size[0] + border, 0),
            max(pad[3] - image.size[1] + border, 0),
        )
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            image = np.pad(np.float32(image), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect")
            h, w, _ = image.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(
                1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]),
            )
            blur = qsize * 0.02
            image += (scipy.ndimage.gaussian_filter(image, [blur, blur, 0]) - image) * np.clip(
                mask * 3.0 + 1.0, 0.0, 1.0
            )
            image += (np.median(image, axis=(0, 1)) - image) * np.clip(mask, 0.0, 1.0)
            image = PIL.Image.fromarray(np.uint8(np.clip(np.rint(image), 0, 255)), "RGB")
            quad += pad[:2]

        # Transform.
        image = image.transform(
            (transform_size, transform_size),
            PIL.Image.QUAD,
            (quad + 0.5).flatten(),
            PIL.Image.BILINEAR,
        )
        if output_size < transform_size:
            image = image.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        imgs.append(image)

    return imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-input_dir",
        type=str,
        default="realpics",
        help="directory with orignal images",
    )
    parser.add_argument("-output_dir", type=str, default="input", help="output directory")
    parser.add_argument(
        "-cache_dir",
        type=str,
        default="cache",
        help="cache directory for model weights (ex: face_68_landmarks.dat )",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    predictor = dlib.shape_predictor("{}/face_68_landmarks.dat".format(args.cache_dir))

    for image_file in Path(args.input_dir).glob("*.*"):
        faces = align_face(str(image_file), predictor)

        for i, face in enumerate(faces):
            face.save(Path(args.output_dir) / (image_file.stem + f"_{i}.png"))
