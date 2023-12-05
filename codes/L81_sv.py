"""
L81 parameterization for multiple source parameter values.
This code is vectorized along source parameters, and takes in a single datum
(vertical profile) at a time. 
"""
import torch
import numpy as np

class Lindzen1981_sv:
    """
    This class can compute multiple instances of L81 for a single profile.
    """
    def __init__(
        self,
        direction=torch.Tensor([torch.pi / 2, torch.pi / 4, 0, -torch.pi / 4]),
        B_m=torch.Tensor([1.0e-3, 1.5e-3, 2.0e-3]),
        c_w: float = 30,
        c_0: float = 0.0,
        c_max: float = 80,
        dc: float = 1.0,
        k: float = 2 * np.pi / 100e3,
    ) -> None:
        """Initialize an L81 instance."""
        self.direction = direction
        self.B_m = B_m.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        self.dc = dc
        self.wave_packet=(self.B_m, c_w, c_0, c_max, self.dc)
        self.h_wavenum = k
        self.phase_speeds, self.mom_flux = self._init_wave_packet(
            self.wave_packet
        )
        self.full_shape = [
            0,
            direction.shape[0],
            B_m.shape[0],
            self.phase_speeds.shape[-1],
        ]
    
    def predict(self, data, split : bool =False):
        return self._loop_predict(data, split)
    
    def _loop_predict(self, datum: dict, split: bool = False) -> dict | torch.Tensor:
        """
        Compute MF profiles for multiple times and locations.
        It is assumed that [u,v,N,rho,MFx,MFy] (attributes of data)
        all have the shape (nlev, ndir, nBm, nphasespeeds).

        Parameters
        ----------
        data: ngc3ICON data structure that contains {
            u : zonal wind,
            v : meridional wind,
            nlev : number of vertical wind levels,
            N : buoyancy frequency,
            source_levs : vertical wind level index that corresponds to 
                          source_lev (default = 600 00) Pa,
            rho : density relative to source level density 
                  (rho[i,j,k] = original_rho[i,j,k] / rho[i,source_lev, k]),
            MFx : zonal momentum flux,
            MFy : meridional momentum flux
            }

        Returns
        -------
        pred: a dict containing keys "MFx" and "MFy".

        """
        self.full_shape[0] = datum["nlev"]
        # Retrieve density at source level.
        rho0 = datum["rho"][datum["source_levs"]]

        # Project winds onto direction.
        wind = (
            (
                torch.cos(self.direction) * datum["u"]
                + torch.sin(self.direction) * datum["v"]
            )
            .unsqueeze(2)
            .unsqueeze(3)
        )

        # Compute unsigned momentum flux profile.
        F_c = (rho0 * self.mom_flux).repeat(1, self.direction.shape[0], 1, 1)

        # Compute signs everywhere and at source level.
        c_hat = torch.sign(self.phase_speeds - wind)
        c_hat0 = c_hat[datum["source_levs"]]

        # Compute breaking conditions.
        breaking_cond = (
            datum["rho"].unsqueeze(2).unsqueeze(3)
            * self.h_wavenum
            * torch.abs(self.phase_speeds - wind) ** 3
            / (2 * datum["N"]).unsqueeze(2).unsqueeze(3)
        )

        # Allocate space for MF.
        MF = torch.zeros(self.full_shape[:-1])

        # Loop over levels (bottom to top).
        sign_change = torch.zeros(
            (1, self.full_shape[1], 1, self.full_shape[3]), dtype=bool
        )
        for lev in range(datum["source_levs"], -1, -1):
            if lev < datum["source_levs"]:
                # Update breaking condition if there was a sign change below,
                # or at current level.
                sign_change = sign_change | (c_hat[lev : lev + 1, :, :, :] != c_hat0)
                breaking_cond[lev : lev + 1, :, :, :] = torch.where(
                    sign_change, 0, breaking_cond[lev : lev + 1, :, :, :]
                )

            # If breaking, (loop is always at or above source).
            breaking = F_c >= breaking_cond[lev : lev + 1, :]

            # Update F_c.
            F_c = torch.where(breaking, breaking_cond[lev : lev + 1, :], F_c)

            # Assign appropriate signs and sum MFs across phase speeds.
            MF[lev] = torch.sum(self.dc * c_hat0 * F_c, dim=3).squeeze()

        if split:
            d = self.direction.unsqueeze(0).unsqueeze(2)
            return {"MFx": torch.cos(d) * MF, "MFy": torch.sin(d) * MF}
        return MF

    def _init_wave_packet(self, wave_packet_info: tuple) -> tuple:
        """
        Compute the initial (at source level) wave packet momentum fluxes.
        (NotationRef: Alexander-Dunkerton 1999)
        First, compute the phase speeds, then apply the Guassian shape.

        Parameters
        ----------
        B_m : max amplitude (amplitude at c_0)
        c_w : half-width of wave packet
        c_0 : wave packet center
        c_max : absolute maximum discretized phase speed
        dc : phase speed discretization

        Returns
        -------
        phase_speeds : linearly spaced phase speeds ranging from -c_max to c_max.
        init_MF : initial (at source level) wave packet momentum fluxes.
        """
        B_m, c_w, c_0, c_max, dc = wave_packet_info
        phase_speeds = self._get_phase_speeds(c_max, dc)
        mom_flux = B_m * torch.exp2(-(((phase_speeds - c_0) / c_w) ** 2))
        return phase_speeds, mom_flux

    @staticmethod
    def _get_phase_speeds(c_max: float, dc: float = 1.0) -> torch.Tensor:
        """
        Compute the GRAVity wave phase spaeeds.

        Parameters
        ----------
        c_max : Norm of the maximum phase speed.
        dc : Phase speed spacing.

        Returns
        -------
        cs : Array of phase speeds.

        """

        return (
            torch.linspace(-c_max, c_max, int(2 * c_max / dc + 1))
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(2)
        )