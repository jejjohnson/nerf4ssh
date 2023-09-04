import pytorch_lightning as pl
import torch
# from eqx_trainer import NumpyLoader, RegressionDataset
from nerf4ssh._src.data.images import load_fox, load_cameraman
from nerf4ssh._src.features.coords import get_image_coordinates
from torch.utils.data import random_split, DataLoader, TensorDataset
from einops import rearrange
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.model_selection import train_test_split


class ImageDM(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        train_size: float = 0.5,
        random_state: int = 123,
        shuffle: bool = False,
        split_method: str = "even",
        resize: int = 1,
        image_url: str = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.split_method = split_method
        self.shuffle = shuffle
        self.train_size = train_size
        self.random_state = random_state
        self.resize = resize
        self.image_url = image_url

    def _load_image(self):
        raise NotImplementedError()

    def load_image(self):
        img = self._load_image()

        if self.resize > 1:
            self.image_height = img.shape[0] // self.resize
            self.image_width = img.shape[1] // self.resize
            img = resize(img, (self.image_height, self.image_width), anti_aliasing=True)
        else:
            self.image_height = img.shape[0]
            self.image_width = img.shape[1]
        return img

    def setup(self, stage=None):

        img = self.load_image()

        coords, pixel_vals = self.image_2_coordinates(img)

        xtrain, xvalid, ytrain, yvalid = self.split(
            coords, pixel_vals, method=self.split_method
        )

        self.ds_train = torch.utils.data.TensorDataset(
            torch.from_numpy(xtrain), torch.from_numpy(ytrain)
        )
        self.ds_valid = torch.utils.data.TensorDataset(
            torch.from_numpy(xvalid), torch.from_numpy(yvalid)
        )
        self.ds_test = torch.utils.data.TensorDataset(
            torch.from_numpy(coords), torch.from_numpy(pixel_vals)
        )

        return self

    def split(self, x, y, method: str = "even"):

        if method == "even":
            xtrain, ytrain = x[::2], y[::2]
            xvalid, yvalid = x[1::2], y[1::2]
        elif method == "random":
            xtrain, xvalid, ytrain, yvalid = train_test_split(
                x, y, random_state=self.random_state, shuffle=True
            )
        else:
            raise ValueError(f"Unrecognized split method")

        return xtrain, xvalid, ytrain, yvalid

    def coordinates_2_image(self, coords):
        return rearrange(
            coords, "(h w) c -> h w c", h=self.image_height, w=self.image_width
        )

    def image_2_coordinates(self, image):
        return get_image_coordinates(image)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                self.ds_train, batch_size=self.batch_size, shuffle=True
            )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
                self.ds_valid, batch_size=self.batch_size, shuffle=False
            )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
                self.ds_test, batch_size=self.batch_size, shuffle=False
            )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
                self.ds_test, batch_size=self.batch_size, shuffle=False
            )


class ImageFox(ImageDM):
    def _load_image(self):
        return load_fox(self.image_url)


class ImageCameraman(ImageDM):
    def _load_image(self):
        return load_cameraman(self.image_url)
