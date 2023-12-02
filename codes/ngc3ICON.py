"""
This script defines a class for Input Output data variables.
Xarray dataset variables are loaded onto memory as torch arrays.
"""
import xarray as xr
import torch
import numpy as np
from utils import (
    find_lowest_true_vectorized,
    find_uppermost_false_vectorized,
    R_DRY,
    C_P,
    GRAV,
)


class ngc3ICON:
    def __init__(
        self,
        dsX: xr.Dataset,
        dsYx: xr.Dataset,
        dsYy: xr.Dataset,
        mask: np.array,
        time_range: list,
        source_lev: float = 200e2,
    ) -> None:
        """
        Initialize an instance of ngc3ICON.

        Attributes:
        -----------
        u : zonal wind
        v : meridional wind
        nlev : number of vertical wind levels
        N : buoyancy frequency
        source_levs : vertical wind level index that corresponds to 
                      source_lev (default = 200 00) Pa.
        rho : density relative to source level density 
              (rho[i,j,k] = original_rho[i,j,k] / rho[i,source_lev, k])
        MFx : zonal momentum flux
        MFy : meridional momentum flux

        [u,v,N,rho,MFx,MFy] all have the shape (time, nlev, cell).

        """

        self.timeidx = self._get_time_range_idx(dsX, time_range)
        self.timevals = dsX.time.isel(time=self.timeidx).values
        self.source_lev = source_lev
        # Load input variables and compute some.
        dsX = dsX.isel(cell=mask, time=self.timeidx)
        self.u = torch.tensor(dsX["ua"].values).unsqueeze(3)
        self.v = torch.tensor(dsX["va"].values).unsqueeze(3)
        self.pfull = torch.tensor(dsX["pfull"].values).unsqueeze(3)
        self.nlev = int(dsX.level_full[-1].data)
        self.N, self.rho, self.source_levs, self.lev_b, self.lev_t = self._get_derived(
            dsX, self.source_lev
        )
        # Load output variables.
        dsYx = dsYx.isel(cell=mask, time=self.timeidx)
        dsYy = dsYy.isel(cell=mask, time=self.timeidx)
        self.MFx = torch.tensor(dsYx["MFx"].values)
        self.MFy = torch.tensor(dsYy["MFy"].values)
        # del dsX dsYx dsYy

    def _get_profile(
        self, timei: int | str | np.datetime64, loci: int | np.ndarray, **kwargs
    ) -> dict:
        latlons = kwargs.get("latlons", None)

        if isinstance(loci, np.ndarray) and latlons is None:
            raise TypeError(
                "If providing lat/lon loc, need latlon information of data slice."
            )

        # Call loaded functions.
        if isinstance(timei, int) is False:
            timei = self._get_timeidx(timei)
        if isinstance(loci, int) is False and latlons is not None:
            loci = self._get_locidx(loci, latlons)
        if isinstance(timei, int | np.int64) and isinstance(loci, int | np.int64):
            return self._get_profile_from_idx(timei, loci)
        raise TypeError("Could not infer time/loc indices.")

    def _get_profile_from_idx(self, timeidx: int, locidx: int) -> dict:
        """
        Picks out a single profile by time and loc index.
        """
        datum = {}
        for var in ["u", "v", "N", "rho", "pfull"]:
            datum[var] = self.__dict__[var][timeidx, :, locidx, :]
        for var in ["MFx", "MFy"]:
            datum[var] = self.__dict__[var][timeidx, :, locidx]
        datum["nlev"] = self.nlev
        datum["lev_t"] = self.lev_t
        datum["lev_b"] = self.lev_b
        datum["source_levs"] = self.source_levs[timeidx, locidx, :]
        return datum

    def _get_timeidx(self, timeidf: str | np.datetime64):
        """
        Get time indices when given as string or np.datetime64.
        """
        if isinstance(timeidf, str):
            slist = timeidf.split("-")
            year, mon, day, hour = (
                int(slist[0]),
                int(slist[1]),
                int(slist[2]),
                int(slist[3][1:]),
            )
            datetimestr = np.datetime64(f"{year}-{mon:02d}-{day:02d}T{hour:02d}:00")
        elif isinstance(timeidf, np.datetime64):
            datetimestr = timeidf
        return np.where(self.timevals == datetimestr)[0][0]

    def _get_locidx(self, locidf: list | np.ndarray, latlons: np.ndarray) -> dict:
        """
        Get loc indices when given as an np.array or list.
        This method requires that we have the latitude longitude information of the data.
        Parameters
        ---------
        locidf = [lat,lon] or np.ndarray([lat,lon]) with shape(2,)
        """
        if isinstance(locidf, list):
            locidf = np.array(locidf)
        locidf = locidf[:, np.newaxis]  # Now has shape (2,1)
        return np.argmin(np.linalg.norm(latlons - locidf, axis=0))

    def _get_derived(self, dsX: xr.Dataset, source_lev: float) -> torch.Tensor:
        """
        Compute derived variables (buoyancy frequency, density, source levels) at each level.

        Parameters
        ----------
        T : Array of temperatures.
        z : Array of heights.

        Returns
        -------
        N : Array of buoyancy frequencies.

        Details
        -------
        Lindzen 1981 uses N^2 = g/T * (dT/dz + g/C_P).
        dT/dz is computed with central differencing for internal levels,
        and bottom/top differencing for the top-most and bottom-most levels.

        """
        # Get top and bottom indices for differencing.
        t_idx = torch.arange(-1, self.nlev - 1)
        t_idx[0] = 0
        b_idx = torch.arange(1, self.nlev + 1)
        b_idx[-1] = self.nlev - 1
        # Compute dTdz
        T = torch.tensor(dsX["ta"].values)
        z = torch.tensor(dsX["zg"].values)
        N = torch.index_select(T, T.shape.index(self.nlev), t_idx) - torch.index_select(
            T, T.shape.index(self.nlev), b_idx
        )
        N /= torch.index_select(
            z, z.shape.index(self.nlev), t_idx
        ) - torch.index_select(z, z.shape.index(self.nlev), b_idx)

        # Compute N^2 = (GRAV / T) * (dTdz + GRAV / C_P).
        N += GRAV / C_P
        N *= GRAV / T
        lev_b, lev_t = (
            torch.min(find_lowest_true_vectorized(N > 0)).item() - 1,
            torch.max(find_uppermost_false_vectorized(N < 0)).item() + 1,
        )
        N = torch.where(N > 2.5e-5, 2.5e-5, N)
        N = torch.sqrt(torch.abs(N))
        # N = (GRAV / T) * (dTdz + GRAV / C_P)
        del z  # , dTdz

        # Compute density (rho).
        P = torch.tensor(dsX["pfull"].values)
        rho = P / (R_DRY * T)

        # Compute source levels from pfull.
        source_levs = self.nlev - torch.sum(torch.Tensor(P >= source_lev), 1)

        # Adjust source level if negative.
        source_levs = torch.where(source_levs > lev_b, lev_b, source_levs)
        del T, P

        return N.unsqueeze(3), rho.unsqueeze(3), source_levs.unsqueeze(2), lev_b, lev_t

    @staticmethod
    def _get_time_range_idx(ds: xr.Dataset, time_range: list) -> range:
        """
        Map date intervals to the correct indices of ds.time.values.
        """
        time_idxs = []
        for timestr, start in zip(time_range, [True, False]):
            slist = timestr.split("-")
            year, mon = int(slist[0]), int(slist[1])
            day = 0 if len(slist) < 3 else int(slist[2])
            hour = 0 if start else 21

            # Find implied day if not indicated.
            if day == 0:
                if start is False:
                    day = 30
                    if mon == 2:
                        if year % 4 == 0:
                            day = 29
                        else:
                            day = 28
                    elif mon in [1, 3, 5, 7, 8, 10, 12]:
                        day = 31
                else:
                    if year == 2020 and mon == 1:
                        day = 20
                    else:
                        day = 1
            hour = 3 if year == 2020 and mon == 1 and day == 20 and start else hour

            # Find index.
            time_idxs.append(
                np.where(
                    ds.get_index("time") == f"{year}-{mon}-{day} {hour:02d}:00:00"
                )[0][0]
            )
        return range(time_idxs[0], time_idxs[1] + 1)
