from typing import List, Dict, Optional, Callable
import warnings
import xarray as xr
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from jejeqx._src.datasets import SpatioTempDataset
from jejeqx._src.dataloaders import NumpyLoader
from jejeqx._src.transforms.spatial import validate_lon, validate_lat, latlon_deg2m
from jejeqx._src.transforms.temporal import decode_cf_time, validate_time, time_rescale
# from jejeqx._src.types.xrdata import (
#     Bounds,
#     Period,
#     TimeAxis,
#     LongitudeAxis,
#     LatitudeAxis,
#     Grid2DT,
# )
from sklearn.model_selection import train_test_split
from dask.array.core import PerformanceWarning
from xarray_dataclasses import asdataset


class AlongTrackDM(pl.LightningDataModule):
    def __init__(
        self,
        paths: List[str],
        spatial_coords: List[str] = ["lat", "lon"],
        temporal_coords: List[str] = ["time"],
        variables: List[str] = ["ssh"],
        batch_size: int = 128,
        select: Dict = None,
        iselect: Dict = None,
        coarsen: Dict = None,
        resample: str = None,
        shuffle: bool = True,
        where_select: Dict = None,
        spatial_transform: Callable = None,
        temporal_transform: Callable = None,
        variable_transform: Callable = None,
        split_seed: int = 123,
        train_size: float = 0.8,
        subset_size: Optional[int] = None,
        subset_seed: int = 42,
        spatial_units: str = "degrees",
        t0: str = "2012-10-01",
        time_freq: int = 1,
        time_unit: str = "seconds",
        evaluation: bool = False,
        decode_times: bool = True,
    ):
        super().__init__()

        self.paths = paths
        self.spatial_coords = spatial_coords
        self.temporal_coords = temporal_coords
        self.variables = variables
        self.batch_size = batch_size
        self.select = select
        self.iselect = iselect
        self.coarsen = coarsen
        self.resample = resample
        self.where_select = where_select
        self.split_seed = split_seed
        self.train_size = train_size
        self.subset_size = subset_size
        self.subset_seed = subset_seed
        self.shuffle = shuffle
        self.spatial_units = spatial_units
        self.t0 = t0
        self.time_freq = time_freq
        self.time_unit = time_unit
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.variable_transform = variable_transform
        self.evaluation = evaluation
        self.decode_times = decode_times

    def load_xrds(self, paths=None, **kwargs):

        if paths is None:
            paths = self.paths

        def preprocess(ds):
            ds = validate_time(ds)
            ds = validate_lon(ds)
            ds = validate_lat(ds)
            ds = ds.sortby("time")

            if self.select is not None:
                ds = ds.sel(**self.select)

            if self.iselect is not None:
                ds = ds.isel(**self.iselect)
            if self.coarsen is not None:
                ds = ds.coarsen(dim=self.coarsen, boundary="trim").mean()
            if self.resample is not None:
                try:
                    ds = ds.resample(time=self.resample).mean()
                except IndexError:
                    pass
            ds = decode_cf_time(ds, units=f"seconds since {self.t0}")

            return ds

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PerformanceWarning)
            # Note: there is an annoying performance memory due to the chunking

            ds = xr.open_mfdataset(
                paths=paths,
                preprocess=preprocess,
                combine="nested",
                concat_dim="time",
                decode_times=self.decode_times,
                **kwargs,
            )

            ds = ds.sortby("time")

            return ds.compute()

    def preprocess(self):

        ds = self.load_xrds(paths=self.paths)

        # rescale space
        if self.spatial_units in ["meters", "metres"]:
            ds = latlon_deg2m(ds, mean=False)

        # rescale time
        ds = time_rescale(
            ds, freq_dt=self.time_freq, freq_unit=self.time_unit, t0=self.t0
        )

        # convert xarray to daraframe
        ds = ds.to_dataframe()

        ds = ds.dropna()

        # extract coordinates (for later)
        self.coord_index = ds.index

        # remove the indexing to get single columns
        ds = ds.reset_index()

        column_names = ds.columns.values

        msg = f"No requested spatial coordinates found in dataset:"
        msg += f"\nTemporal Coords: {self.spatial_coords}"
        msg += f"\nColumns: {column_names}"
        assert len(set(self.spatial_coords).intersection(column_names)) == len(
            self.spatial_coords
        ), msg

        msg = f"No requested temporal coordinates found in dataset:"
        msg += f"\nTemporal Coords: {self.temporal_coords}"
        msg += f"\nColumns: {column_names}"
        assert len(set(self.temporal_coords).intersection(column_names)) == len(
            self.temporal_coords
        ), msg

        msg = f"No requested variables found in dataset:"
        msg += f"\nVariables: {self.variables}"
        msg += f"\nColumns: {column_names}"
        assert len(set(self.variables).intersection(column_names)) == len(
            self.variables
        ), msg

        x = ds[self.spatial_coords]
        t = ds[self.temporal_coords]
        y = ds[self.variables]

        # do specific spatial-temporal-variable transformations
        if self.spatial_transform is not None:
            if not self.evaluation:
                x = self.spatial_transform.fit_transform(x)
            else:
                x = self.spatial_transform.transform(x)

        if self.temporal_transform is not None:
            if not self.evaluation:
                t = self.temporal_transform.fit_transform(t)
            else:
                t = self.temporal_transform.transform(t)
        if self.variable_transform is not None:
            if not self.evaluation:
                y = self.variable_transform.fit_transform(y)
            else:
                y = self.variable_transform.transform(y)

        # extract the values
        x, t, y = x.values, t.values, y.values

        self.spatial_dims = x.shape[-1]
        self.temporal_dims = t.shape[-1]
        self.variable_dims = y.shape[-1]

        return x, t, y

    def setup(self, stage=None):

        x, t, y = self.preprocess()

        self.ds_test = SpatioTempDataset(spatial_coords=x, temporal_coords=t, data=y)

        if self.subset_size is not None:
            x, t, y = self.subset(x, t, y)

        # train/validation/test split
        xtrain, xvalid, ttrain, tvalid, ytrain, yvalid = self.split(x, t, y)

        # create spatial-temporal datasets
        self.ds_train = SpatioTempDataset(
            spatial_coords=xtrain, temporal_coords=ttrain, data=ytrain
        )
        self.ds_valid = SpatioTempDataset(
            spatial_coords=xvalid, temporal_coords=tvalid, data=yvalid
        )

    def subset(self, x, t, y):

        x, _, t, _, y, _ = train_test_split(
            x,
            t,
            y,
            train_size=self.subset_size,
            random_state=self.subset_seed,
            shuffle=True,
        )

        return x, t, y

    def split(self, x, t, y):

        xtrain, xvalid, ttrain, tvalid, ytrain, yvalid = train_test_split(
            x,
            t,
            y,
            train_size=self.train_size,
            random_state=self.split_seed,
            shuffle=True,
        )
        return xtrain, xvalid, ttrain, tvalid, ytrain, yvalid

    def data_to_df(self, x):
        return pd.DataFrame(x, index=self.coord_index, columns=self.variables)

    def train_dataloader(self):
        return NumpyLoader(
            self.ds_train, batch_size=self.batch_size, shuffle=self.shuffle
        )

    def val_dataloader(self):
        return NumpyLoader(self.ds_valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return NumpyLoader(self.ds_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return NumpyLoader(self.ds_test, batch_size=self.batch_size)


class EvalCoordDM(AlongTrackDM):
    def setup(self, stage=None):

        x, t, y = self.preprocess()

        self.ds_test = SpatioTempDataset(spatial_coords=x, temporal_coords=y, data=y)

    def train_dataloader(self):
        raise NotImplementedError()

    def val_dataloader(self):
        raise NotImplementedError()

    def test_dataloader(self):
        return NumpyLoader(self.ds_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return NumpyLoader(self.ds_test, batch_size=self.batch_size)


# class EvalGridDM(pl.LightningDataModule):
#     def __init__(
#         self,
#         lon_limits: Bounds,
#         lat_limits: Bounds,
#         time_limits: Period,
#         spatial_coords: List[str] = ["lat", "lon"],
#         temporal_coords: List[str] = ["time"],
#         batch_size: int = 128,
#         spatial_units: str = "degrees",
#         t0: str = "2012-10-01",
#         time_freq: int = 1,
#         time_unit: str = "seconds",
#         spatial_transform: Callable = None,
#         temporal_transform: Callable = None,
#         variable_transform: Callable = None,
#     ):
#         super().__init__()

#         self.lon_limits = lon_limits
#         self.lat_limits = lat_limits
#         self.time_limits = time_limits
#         self.spatial_coords = spatial_coords
#         self.temporal_coords = temporal_coords
#         self.batch_size = batch_size
#         self.spatial_transform = spatial_transform
#         self.temporal_transform = temporal_transform
#         self.variable_transform = variable_transform
#         self.spatial_units = spatial_units
#         self.t0 = t0
#         self.time_freq = time_freq
#         self.time_unit = time_unit

#     def preprocess(self):

#         # create spatialtemporal grid
#         time_axis = TimeAxis.init_from_limits(
#             t_min=pd.to_datetime(self.time_limits.t_min),
#             t_max=pd.to_datetime(self.time_limits.t_max),
#             dt=pd.to_timedelta(self.time_limits.dt_freq, self.time_limits.dt_unit),
#         )
#         lon_axis = LongitudeAxis.init_from_limits(
#             lon_min=self.lon_limits.val_min,
#             lon_max=self.lon_limits.val_max,
#             dlon=self.lon_limits.val_step,
#         )
#         lat_axis = LatitudeAxis.init_from_limits(
#             lat_min=self.lat_limits.val_min,
#             lat_max=self.lat_limits.val_max,
#             dlat=self.lat_limits.val_step,
#         )

#         data = np.empty((time_axis.ndim, lat_axis.ndim, lon_axis.ndim))

#         ds = Grid2DT(
#             data=data, time=time_axis, lat=lat_axis, lon=lon_axis, name="empty"
#         )

#         ds = asdataset(ds)

#         ds = validate_time(ds)
#         ds = validate_lon(ds)
#         ds = validate_lat(ds)
#         ds = decode_cf_time(ds, units=f"seconds since {self.t0}")

#         # rescale space
#         if self.spatial_units in ["meters", "metres"]:
#             ds = latlon_deg2m(ds, mean=False)

#         # rescale time
#         ds = time_rescale(
#             ds, freq_dt=self.time_freq, freq_unit=self.time_unit, t0=self.t0
#         )

#         # convert xarray to daraframe
#         ds = ds.to_dataframe()

#         ds = ds.dropna()

#         # extract coordinates (for later)
    #     self.coord_index = ds.index

    #     # remove the indexing to get single columns
    #     ds = ds.reset_index()

    #     column_names = ds.columns.values

    #     msg = f"No requested spatial coordinates found in dataset:"
    #     msg += f"\nTemporal Coords: {self.spatial_coords}"
    #     msg += f"\nColumns: {column_names}"
    #     assert len(set(self.spatial_coords).intersection(column_names)) == len(
    #         self.spatial_coords
    #     ), msg

    #     msg = f"No requested temporal coordinates found in dataset:"
    #     msg += f"\nTemporal Coords: {self.temporal_coords}"
    #     msg += f"\nColumns: {column_names}"
    #     assert len(set(self.temporal_coords).intersection(column_names)) == len(
    #         self.temporal_coords
    #     ), msg

    #     x = ds[self.spatial_coords]
    #     t = ds[self.temporal_coords]

    #     # do specific spatial-temporal-variable transformations
    #     if self.spatial_transform is not None:
    #         x = self.spatial_transform.transform(x)
    #     if self.temporal_transform is not None:
    #         # print(t.min(), t.max())
    #         t = self.temporal_transform.transform(t)
    #         # print(t.min(), t.max())

    #     # extract the values
    #     x, t = x.values, t.values

    #     self.spatial_dims = x.shape[-1]
    #     self.temporal_dims = t.shape[-1]

    #     return x, t

    # def setup(self, stage=None):

    #     x, t = self.preprocess()

    #     self.ds_predict = SpatioTempDataset(spatial_coords=x, temporal_coords=t)

    # def data_to_df(self, x, names=["ssh"]):
    #     return pd.DataFrame(x, index=self.coord_index, columns=names)

    # def train_dataloader(self):
    #     raise NotImplementedError()

    # def val_dataloader(self):
    #     raise NotImplementedError()

    # def test_dataloader(self):
    #     raise NotImplementedError()

    # def predict_dataloader(self):
    #     return NumpyLoader(self.ds_predict, batch_size=self.batch_size)
