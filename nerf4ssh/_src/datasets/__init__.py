import torch.utils.data as data


class RegressionDataset(data.Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SpatioTempDataset(data.Dataset):
    def __init__(
        self, spatial_coords=None, temporal_coords=None, params=None, data=None
    ):
        super().__init__()
        self.spatial_coords = spatial_coords
        self.temporal_coords = temporal_coords
        self.params = params
        self.data = data

    def __len__(self):
        return self.spatial_coords.shape[0]

    def __getitem__(self, idx):
        outputs = dict()
        if self.spatial_coords is not None:
            outputs["spatial"] = self.spatial_coords[idx]
        if self.temporal_coords is not None:
            outputs["temporal"] = self.temporal_coords[idx]
        if self.params is not None:
            outputs["params"] = self.params[idx]
        if self.data is not None:
            outputs["data"] = self.data[idx]

        return outputs


class SpatioTempParamDataset(data.Dataset):
    def __init__(self, x_space, x_time, x_params, y):
        super().__init__()
        self.x_space = x_space
        self.x_time = x_time
        self.x_params = x_params
        self.y = y

    def __len__(self):
        return self.x_space.shape[0]

    def __getitem__(self, idx):
        outputs = dict()

        outputs["spatial"] = self.x_space[idx]
        outputs["temporal"] = self.x_time[idx]
        outputs["params"] = self.x_params[idx]
        outputs["y"] = self.y[idx]

        return outputs
