import autoroot
from typing import List
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import xarray as xr
import cmocean as cmo
import numpy as np
import jejeqx._src.transforms.xarray.geostrophic as geocalc
from jejeqx._src.transforms.xarray.grid import latlon_deg2m, time_rescale
from jejeqx._src.transforms.xarray.psd import (
    psd_spacetime,
    psd_isotropic,
    psd_average_freq,
)
from jejeqx._src.metrics.xarray.psd import (
    psd_isotropic_resolved_scale,
    psd_isotropic_score,
    psd_spacetime_resolved_scale,
    psd_spacetime_score,
)

from jejeqx._src.viz.xarray.psd import plot_psd_isotropic, plot_psd_spacetime_wavelength
from jejeqx._src.viz.xarray.psd_score import (
    plot_psd_isotropic_score,
    plot_psd_spacetime_score_wavelength,
)
from jejeqx._src.viz.utils import get_cbar_label

def calculate_anomaly(ds, variable="ssh", dim=["lat", "lon"]):
    ds[f"{variable}_anomaly"] = ds[variable] - ds[variable].mean(dim=dim)
    return ds


def calculate_physical_quantities(da: xr.DataArray) -> xr.Dataset:

    # SSH
    ds = geocalc.get_ssh_dataset(da)

    # Stream Function
    ds = geocalc.calculate_streamfunction(ds, "ssh")

    # U,V Velocities
    ds = geocalc.calculate_velocities_sf(ds, "psi")

    # Kinetic Energy
    ds = geocalc.calculate_kinetic_energy(ds, ["u", "v"])

    # Relative Vorticity
    ds = geocalc.calculate_relative_vorticity_uv(ds, ["u", "v"], normalized=True)

    # Strain
    ds = geocalc.calculate_strain_magnitude(ds, ["u", "v"], normalized=True)

    # Okubo-Weiss
    ds = geocalc.calculate_okubo_weiss(ds, ["u", "v"], normalized=True)

    return ds


def calculate_spacetime_psd(ds, freq_dt=1, freq_unit="D"):

    ds = latlon_deg2m(ds, mean=True)
    ds = time_rescale(ds, freq_dt, freq_unit)

    variables = ["ssh", "u", "v", "ke", "vort_r", "strain", "ow"]

    ds_psd = xr.Dataset()

    # calculate spacetime PSDs
    for ivariable in tqdm(variables):
        ds_psd[ivariable] = psd_average_freq(
            psd_spacetime(ds[ivariable], ["time", "lon"])
        )

    return ds_psd


def calculate_isotropic_psd(ds, freq_dt=1, freq_unit="D"):

    ds = latlon_deg2m(ds, mean=True)
    ds = time_rescale(ds, freq_dt, freq_unit)

    variables = ["ssh", "u", "v", "ke", "vort_r", "strain", "ow"]

    # calculate isotropic PSDs
    ds_psd = xr.Dataset()

    for ivariable in tqdm(variables):
        ds_psd[ivariable] = psd_average_freq(
            psd_isotropic(ds[ivariable], ["lat", "lon"])
        )

    return ds_psd


def calculate_isotropic_psd_score(ds, ds_ref):

    freq_dt = 1
    freq_unit = "D"

    ds = latlon_deg2m(ds, mean=True)
    ds = time_rescale(ds, freq_dt, freq_unit)

    ds_ref = latlon_deg2m(ds_ref, mean=True)
    ds_ref = time_rescale(ds_ref, freq_dt, freq_unit)

    variables = [
        "ssh",
        "ke",
        "vort_r",
        "strain",
    ]
    dims = ["lat", "lon"]

    ds_psd_score = xr.Dataset()

    for ivariable in tqdm(variables):
        # calculate isotropic psd score
        ds_psd_score[ivariable] = psd_isotropic_score(
            ds[ivariable], ds_ref[ivariable], dims=dims
        )
        ds_psd_score[ivariable] = psd_isotropic_resolved_scale(ds_psd_score[ivariable])

    return ds_psd_score


def calculate_spacetime_psd_score(ds, ds_ref):

    freq_dt = 1
    freq_unit = "D"

    ds = latlon_deg2m(ds, mean=True)
    ds = time_rescale(ds, freq_dt, freq_unit)

    ds_ref = latlon_deg2m(ds_ref, mean=True)
    ds_ref = time_rescale(ds_ref, freq_dt, freq_unit)

    variables = ["ssh", "ke", "vort_r", "strain"]
    dims = ["time", "lon"]

    ds_psd_score = xr.Dataset()

    for ivariable in tqdm(variables):
        # calculate isotropic psd score
        ds_psd_score[ivariable] = psd_spacetime_score(
            ds[ivariable], ds_ref[ivariable], dims=dims
        )
        ds_psd_score[ivariable] = psd_spacetime_resolved_scale(ds_psd_score[ivariable])

    return ds_psd_score


def plot_analysis_vars(ds: List[xr.Dataset], names: List[str] = None, figsize=None):

    if figsize is None:
        figsize = (12, 20)

    ncols = len(ds)

    fig, ax = plt.subplots(nrows=7, ncols=ncols, figsize=figsize)

    # SSH
    vmin = np.min([ids.ssh.min() for ids in ds])
    vmax = np.max([ids.ssh.max() for ids in ds])

    cond = lambda x: isinstance(x, np.ndarray)
    for iax, ids in zip(ax[0] if cond(ax[0]) else [ax[0]], ds):
        cbar_kwargs = {"label": get_cbar_label(ids.ssh)}
        ids.ssh.plot.pcolormesh(
            ax=iax, cmap="viridis", vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs
        )
        iax.set_aspect("equal", "box")

    # U
    vmin = np.min([ids.u.min() for ids in ds])
    vmax = np.max([ids.u.max() for ids in ds])
    for iax, ids in zip(ax[1] if cond(ax[1]) else [ax[1]], ds):
        cbar_kwargs = {"label": get_cbar_label(ids.u)}
        ids.u.plot.pcolormesh(
            ax=iax, cmap="gray", vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs
        )
        iax.set_aspect("equal", "box")

    # v
    vmin = np.min([ids.v.min() for ids in ds])
    vmax = np.max([ids.v.max() for ids in ds])
    for iax, ids in zip(ax[2] if cond(ax[2]) else [ax[2]], ds):
        cbar_kwargs = {"label": get_cbar_label(ids.v)}
        ids.v.plot.pcolormesh(
            ax=iax, cmap="gray", vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs
        )
        iax.set_aspect("equal", "box")

    # Kinetic Energy
    vmin = np.min([ids.ke.min() for ids in ds])
    vmax = np.max([ids.ke.max() for ids in ds])
    for iax, ids in zip(ax[3] if cond(ax[3]) else [ax[3]], ds):
        cbar_kwargs = {"label": get_cbar_label(ids.ke)}
        ids.ke.plot.pcolormesh(
            ax=iax,
            cmap="YlGnBu_r",  # vmin=vmin, vmax=vmax,
            robust=True,
            cbar_kwargs=cbar_kwargs,
        )
        iax.set_aspect("equal", "box")

    # Relative Vorticity
    vmin = np.min([ids.vort_r.min() for ids in ds])
    vmax = np.max([ids.vort_r.max() for ids in ds])
    for iax, ids in zip(ax[4] if cond(ax[4]) else [ax[4]], ds):
        cbar_kwargs = {"label": get_cbar_label(ids.vort_r)}
        ids.vort_r.plot.pcolormesh(
            ax=iax, cmap="RdBu_r", cbar_kwargs=cbar_kwargs  # vmin=vmin, vmax=vmax,
        )
        iax.set_aspect("equal", "box")

    # STRAIN
    vmin = 0.001 * np.min([ids.strain.min() for ids in ds])
    vmax = 0.995 * np.max([ids.strain.max() for ids in ds])
    for iax, ids in zip(ax[5] if cond(ax[5]) else [ax[5]], ds):
        cbar_kwargs = {"label": get_cbar_label(ids.strain)}
        ids.strain.plot.pcolormesh(
            ax=iax,
            cmap=cmo.cm.speed,  # vmin=vmin, vmax=vmax,
            robust=True,
            cbar_kwargs=cbar_kwargs,
        )
        iax.set_aspect("equal", "box")

    # Okubo-Weiss
    vmin = np.min([ids.ow.min() for ids in ds])
    vmax = np.max([ids.ow.max() for ids in ds])
    for iax, ids in zip(ax[6] if cond(ax[6]) else [ax[6]], ds):
        cbar_kwargs = {"label": get_cbar_label(ids.ow)}
        ids.ow.plot.contourf(
            ax=iax, cmap="cividis", vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs
        )
        iax.set_aspect("equal", "box")

    if names is not None:
        fig.suptitle(t=names)

    plt.tight_layout()
    return fig, ax


def plot_analysis_psd_iso(ds: List[xr.Dataset], names: List[str]):

    ncols = len(ds)

    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(5, 15))

    # SSH
    for iname, ids in zip(names, ds):
        scale = "km"
        units = "$m^{2}$/cycles/m"
        plot_psd_isotropic(ids.ssh, units=units, scale=scale, ax=ax[0], label=iname)

    ## U
    # for iname, ids in zip(names, ds):
    #    scale = "km"
    #    units = "U-Velocity"
    #    plot_psd_isotropic(ids.u, units=units, scale=scale, ax=ax[1], label=iname)

    # # v
    # for iname, ids in zip(names, ds):
    #    scale = "km"
    #    units = "V-Velocity"
    #    plot_psd_isotropic(ids.v, units=units, scale=scale, ax=ax[2], label=iname)

    # Kinetic Energy
    for iname, ids in zip(names, ds):
        scale = "km"
        units = "Kinetic Energy"
        plot_psd_isotropic(ids.ke, units=units, scale=scale, ax=ax[1], label=iname)

    # Relative Vorticity
    for iname, ids in zip(names, ds):
        scale = "km"
        units = "Relative Vorticity"
        plot_psd_isotropic(ids.vort_r, units=units, scale=scale, ax=ax[2], label=iname)

    # STRAIN
    for iname, ids in zip(names, ds):
        scale = "km"
        units = "Strain"
        plot_psd_isotropic(ids.strain, units=units, scale=scale, ax=ax[3], label=iname)

    ## STRAIN
    # for iname, ids in zip(names, ds):
    #    scale = "km"
    #    units = "Okubo-Weiss"
    #   plot_psd_isotropic(ids.ow, units=units, scale=scale, ax=ax[6], label=iname)

    plt.tight_layout()
    return fig, ax


def plot_analysis_psd_iso_score(
    ds: List[xr.Dataset],
    names: List[str],
    colors: List[str],
):

    ncols = len(ds)

    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(5, 15))

    # SSH
    labels = []
    for iname, icolor, ids in zip(names, colors, ds):
        scale = "km"
        units = "SSH"
        _, ax[0], _ = plot_psd_isotropic_score(
            ids.ssh, scale=scale, ax=ax[0], name=iname, color=icolor
        )
        plt.legend()

    # Kinetic Energy
    labels = []
    for iname, icolor, ids in zip(names, colors, ds):
        scale = "km"
        units = "U-Velocity"
        _, ax[1], _ = plot_psd_isotropic_score(
            ids.ke, scale=scale, ax=ax[1], name=iname, color=icolor
        )

    # Relative Vorticity
    labels = []
    for iname, icolor, ids in zip(names, colors, ds):
        scale = "km"
        units = "Relative Vorticity"
        _, ax[2], _ = plot_psd_isotropic_score(
            ids.vort_r, scale=scale, ax=ax[2], name=iname, color=icolor
        )

    # Strain
    labels = []
    for iname, icolor, ids in zip(names, colors, ds):
        scale = "km"
        units = "Relative Vorticity"
        _, ax[3], _ = plot_psd_isotropic_score(
            ids.strain, scale=scale, ax=ax[3], name=iname, color=icolor
        )

    plt.tight_layout()
    return fig, ax


def plot_analysis_psd_spacetime(ds: List[xr.Dataset], names: List[str]):

    ncols = len(ds)

    fig, ax = plt.subplots(nrows=4, ncols=ncols, figsize=(12, 20))

    # SSH
    cond = lambda x: isinstance(x, np.ndarray)
    for iax, ids in zip(ax[0] if cond(ax[0]) else [ax[0]], ds):
        scale = "km"
        units = "SSH"  # "$m^{2}$/cycles/m"
        plot_psd_spacetime_wavelength(
            ids.ssh,
            psd_units=units,
            space_scale=scale,
            ax=iax,
        )
        iax.set_aspect("equal", "box")

    # Kinetic Energy
    for iax, ids in zip(ax[1] if cond(ax[1]) else [ax[1]], ds):
        scale = "km"
        units = "Kinetic Energy"
        plot_psd_spacetime_wavelength(
            ids.ke,
            psd_units=units,
            space_scale=scale,
            ax=iax,
        )
        iax.set_aspect("equal", "box")

    # Relative Vorticity
    for iax, ids in zip(ax[2] if cond(ax[2]) else [ax[2]], ds):
        scale = "km"
        units = "Relative Vorticity"
        plot_psd_spacetime_wavelength(
            ids.vort_r,
            psd_units=units,
            space_scale=scale,
            ax=iax,
        )
        iax.set_aspect("equal", "box")

    # STRAIN
    for iax, ids in zip(ax[3] if cond(ax[3]) else [ax[3]], ds):
        scale = "km"
        units = "Strain"
        plot_psd_spacetime_wavelength(
            ids.strain,
            psd_units=units,
            space_scale=scale,
            ax=iax,
        )
        iax.set_aspect("equal", "box")

    plt.tight_layout()
    return fig, ax


def plot_analysis_psd_spacetime_score(ds: List[xr.Dataset], names: List[str]):

    ncols = len(ds)

    fig, ax = plt.subplots(nrows=4, ncols=ncols, figsize=(12, 20))

    # SSH
    for iax, ids in zip(ax[0] if isinstance(ax[0], list) else [ax[0]], ds):
        scale = "km"
        units = "SSH"  # "$m^{2}$/cycles/m"
        plot_psd_spacetime_score_wavelength(
            ids.ssh,
            psd_units=units,
            space_scale=scale,
            ax=iax,
        )
        iax.set_aspect("equal", "box")

    # Kinetic Energy
    for iax, ids in zip(ax[1] if isinstance(ax[1], list) else [ax[1]], ds):
        scale = "km"
        units = "Kinetic Energy"
        plot_psd_spacetime_score_wavelength(
            ids.ke,
            psd_units=units,
            space_scale=scale,
            ax=iax,
        )
        iax.set_aspect("equal", "box")

    # Relative Vorticity
    ax2 = ax[2] if isinstance(ax[2], list) else [ax[2]]
    for iax, ids in zip(ax2, ds):
        scale = "km"
        units = "Relative Vorticity"
        plot_psd_spacetime_score_wavelength(
            ids.vort_r,
            psd_units=units,
            space_scale=scale,
            ax=iax,
        )
        iax.set_aspect("equal", "box")

    # STRAIN
    ax3 = ax[3] if isinstance(ax[3], list) else [ax[3]]
    for iax, ids in zip(ax3, ds):
        scale = "km"
        units = "Strain"
        plot_psd_spacetime_score_wavelength(
            ids.strain,
            psd_units=units,
            space_scale=scale,
            ax=iax,
        )
        iax.set_aspect("equal", "box")

    plt.tight_layout()
    return fig, ax
