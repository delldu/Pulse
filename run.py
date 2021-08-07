from PULSE import PULSE
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from pathlib import Path
from PIL import Image
import torchvision
from math import log10, ceil
import argparse
import pdb


class Images(Dataset):
    def __init__(self, root_dir, duplicates):
        self.root_path = Path(root_dir)
        self.image_list = list(self.root_path.glob("*.png"))
        self.duplicates = duplicates  # Number of times to duplicate the image in the dataset to produce multiple HR images
        # root_dir = 'input'
        # duplicates = 1
        # (Pdb) pp self.image_list
        # [PosixPath('input/Patrice_Chereau_0002_0.png'),
        #  PosixPath('input/Oswaldo_Paya_0005_0.png')]

    def __len__(self):
        return self.duplicates * len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx // self.duplicates]
        image = torchvision.transforms.ToTensor()(Image.open(img_path))
        # (Pdb) self.image_list[0].stem
        # 'Patrice_Chereau_0002_0'

        if self.duplicates == 1:
            return image, img_path.stem
        else:
            return image, img_path.stem + f"_{(idx % self.duplicates)+1}"


parser = argparse.ArgumentParser(description="PULSE")

# I/O arguments
parser.add_argument(
    "-input_dir", type=str, default="input", help="input data directory"
)
parser.add_argument(
    "-output_dir", type=str, default="output", help="output data directory"
)
parser.add_argument(
    "-cache_dir", type=str, default="cache", help="cache directory for model weights"
)
parser.add_argument(
    "-duplicates",
    type=int,
    default=3,
    help="How many HR images to produce for every image in the input directory",
)
parser.add_argument(
    "-batch_size", type=int, default=1, help="Batch size to use during optimization"
)

# PULSE arguments
parser.add_argument("-seed", type=int, help="manual seed to use")
parser.add_argument(
    "-loss_str", type=str, default="100*L2+0.05*GEOCROSS", help="Loss function to use"
)
parser.add_argument(
    "-eps", type=float, default=5e-3, help="Target for downscaling loss (L2)"
)
parser.add_argument(
    "-noise_type", type=str, default="trainable", help="zero, fixed, or trainable"
)
parser.add_argument(
    "-num_trainable_noise_layers",
    type=int,
    default=5,
    help="Number of noise layers to optimize",
)
parser.add_argument(
    "-tile_latent",
    action="store_true",
    help="Whether to forcibly tile the same latent 18 times",
)
parser.add_argument(
    "-bad_noise_layers",
    type=str,
    default="17",
    help="List of noise layers to zero out to improve image quality",
)
parser.add_argument(
    "-opt_name",
    type=str,
    default="adam",
    help="Optimizer to use in projected gradient descent",
)
parser.add_argument(
    "-learning_rate",
    type=float,
    default=0.4,
    help="Learning rate to use during optimization",
)
parser.add_argument(
    "-steps", type=int, default=100, help="Number of optimization steps"
)
parser.add_argument(
    "-lr_schedule",
    type=str,
    default="linear1cycledrop",
    help="fixed, linear1cycledrop, linear1cycle",
)

kwargs = vars(parser.parse_args())

dataset = Images(kwargs["input_dir"], duplicates=kwargs["duplicates"])
out_path = Path(kwargs["output_dir"])
out_path.mkdir(parents=True, exist_ok=True)

dataloader = DataLoader(dataset, batch_size=kwargs["batch_size"])

model = PULSE(cache_dir=kwargs["cache_dir"])
model = DataParallel(model)

toPIL = torchvision.transforms.ToPILImage()

for ref_im, ref_im_name in dataloader:
    print(ref_im_name)

    out_im = model(ref_im, **kwargs)
    for j, (HR, LR) in enumerate(model(ref_im, **kwargs)):
        for i in range(kwargs["batch_size"]):
            output_filename = out_path / f"{ref_im_name[i]}.png"
            toPIL(HR[i].cpu().detach().clamp(0, 1)).save(output_filename)
