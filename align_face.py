import glob
import dlib
from pathlib import Path
import argparse
from bicubic import BicubicDownSample
import torchvision
from shape_predictor import align_face
import pdb

parser = argparse.ArgumentParser(description="PULSE")

parser.add_argument(
    "-input_dir", type=str, default="realpics", help="directory with unprocessed images"
)
parser.add_argument("-output_dir", type=str, default="input", help="output directory")
parser.add_argument(
    "-output_size",
    type=int,
    default=64,
    help="size to downscale the input images to, must be power of 2",
)
parser.add_argument(
    "-cache_dir", type=str, default="cache", help="cache directory for model weights"
)

args = parser.parse_args()

cache_dir = Path(args.cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

predictor = dlib.shape_predictor("cache/face_68_landmarks.dat")

for im in Path(args.input_dir).glob("*.*"):
    faces = align_face(str(im), predictor)
    for i, face in enumerate(faces):
        if args.output_size:
            factor = 1024 // args.output_size
            # (Pdb) factor 16
            assert args.output_size * factor == 1024
            D = BicubicDownSample(factor=factor)

            face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0).cuda()
            face_tensor_lr = D(face_tensor)[0].cpu().detach().clamp(0, 1)
            face = torchvision.transforms.ToPILImage()(face_tensor_lr)

        face.save(Path(args.output_dir) / (im.stem + f"_{i}.png"))
