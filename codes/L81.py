"""
Vectorized and single profile implementation of the L81 parameterization.
"""
import torch
from ngc3ICON import ngc3ICON as n3
import numpy as np

R_DRY = 287.04
C_P = 7 * R_DRY / 2
GRAV = 9.8

class Lindzen1981:
    def __init__(self, 
        direction: str = "NE", 
        B_m: float = 1.5e-3, 
        c_w: float = 30,
        c_0: float = 0.0,
        c_max: float = 80,
        dc: float = 1.0,
        k: float = 2 * np.pi / 100e3
        ) -> None:
        """Initialize an L81 instance."""
        if isinstance(direction, str):
            dir_dict={"N": (0,1),
                      "NE": (1/np.sqrt(2),1/np.sqrt(2)),
                      "E": (1,0),
                      "SE":  (1/np.sqrt(2),-1/np.sqrt(2))
                     }
            self.direction = dir_dict[direction]
        elif isinstance(direction, float):
            self.direction=(np.cos(direction), np.sin(direction))
        self.B_m = B_m
        self.c_w = c_w
        self.c_0 = c_0
        self.c_max = c_max
        self.dc = dc
        
        self.h_wavenum = k
        self.phase_speeds, self.mom_flux = self._init_wave_packet((B_m, c_w, c_0, c_max, dc))
    
    def _predict(self, datum: dict, split:bool=False) -> dict:
        """
        Compute a single MF profile. 
        datum values are the same parameters as _vectorized_predict. 
        """
        # Retrieve density at source level.
        rho0 = datum["rho"][datum["source_levs"]]
        
        # Project winds onto direction.
        wind = (self.direction[0]*datum["u"]) + (self.direction[1]*datum["v"])
        
        # Compute unsigned momentum flux profile.
        F_c = rho0 * self.mom_flux
        
        # Compute signs everywhere and at source level.
        c_hat = torch.sign(self.phase_speeds-wind)
        c_hat0 = c_hat[datum["source_levs"]]
        
        # Compute breaking conditions.
        breaking_cond = datum["rho"] * self.h_wavenum*torch.abs(self.phase_speeds - wind)**3 / (2*datum["N"])
        
        # Create mask for being above source level.
        above_source = torch.Tensor(range(90)) <= datum["source_levs"]
        
        # Update breaking conditions to 0 if there's a sign change. 
        breaking_cond = torch.where(c_hat == c_hat0, breaking_cond,0)
        
        # Allocate space for MF. 
        MF = torch.zeros(datum["MFx"].shape)
        
        # Loop over levels (bottom to top). 
        # If breaking condition met and above source level, adjust F_c.
        # Before moving on to next level, sum up the fluxes. 
        for lev in range(datum["nlev"]-1,-1,-1):
            # If breaking,
            breaking= F_c >= breaking_cond[lev:lev+1,:]

            # and above source, then update. update is a mask.
            update = breaking * above_source[lev:lev+1]

            # This is equivalent to assigning breaking_cond where update is True, 
            # and F_c to where update is False.
            # Remember that breaking_cond was includes sign change already. 
            F_c = torch.where(update, breaking_cond[lev:lev+1,:], F_c)

            # Assign appropriate signs and sum MFs across phase speeds.
            MF[lev] = torch.sum(c_hat0*F_c*above_source,dim=1).squeeze()
            
        if split:
            MFx, MFy = self.direction[0]*MF, self.direction[1]*MF
            MF = {"MFx":MFx,"MFy":MFy}
        return MF
        
    
    def _vectorized_predict(self, data:n3, split:bool = False) -> np.ndarray:
        """
        Compute MF profiles for multiple times and locations.
        It is assumed that [u,v,N,rho,MFx,MFy] (attributes of data)
        all have the shape (time, nlev, cell).

        Parameters
        ----------
        data: ngc3ICON data structure that contains {     
            u : zonal wind,
            v : meridional wind,
            nlev : number of vertical wind levels,
            N : buoyancy frequency,
            source_levs : vertical wind level index that corresponds to source_lev (default = 600 00) Pa,
            rho : density relative to source level density (rho[i,j,k] = original_rho[i,j,k] / rho[i,source_lev, k]),
            MFx : zonal momentum flux,
            MFy : meridional momentum flux
            }
            
        Returns 
        -------
        pred: a dict containing keys "MFx" and "MFy".

        """
        # Retrieve density at source level.
        rho0=torch.gather(data.rho, 1, data.source_levs.unsqueeze(1))

        # Project winds onto direction.
        wind = (self.direction[0] * data.u) + (self.direction[1] * data.v)
        
        # Compute unsigned momentum flux profile.
        F_c = rho0*self.mom_flux
        
        # Compute signs everywhere and at source level.
        c_hat = torch.sign(self.phase_speeds-wind)
        c_hat0 = torch.gather(c_hat, 1, data.source_levs.unsqueeze(1))
        
        # Compute breaking conditions.
        breaking_cond = data.rho * self.h_wavenum*torch.abs(self.phase_speeds - wind)**3 / (2*data.N)
        
        # Update breaking conditions to 0 if there's a sign change. 
        breaking_cond = torch.where(c_hat == c_hat0, breaking_cond,0)

        # Create mask for being above source level.
        levs = torch.Tensor(range(90)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        above_source = levs <= data.source_levs.unsqueeze(1)
        
        # Allocate space for MF. 
        MF = torch.zeros(data.__getattribute__('MFx').shape)
        
        # Loop over levels (bottom totop). 
        # If breaking condition met and above source level, adjust F_c.
        # Before moving on to next level, sum up the fluxes. 
        for lev in range(data.nlev-1,-1,-1):
            # If breaking,
            breaking= F_c >= breaking_cond[:,lev:lev+1,:,:]

            # and above source, then update. update is a mask.
            update = breaking * above_source[:,lev:lev+1,:,:]

            # This is equivalent to assigning breaking_cond where update is True, 
            # and F_c to where update is False.
            # Remember that breaking_cond was includes sign change already. 
            F_c = torch.where(update, breaking_cond[:,lev:lev+1,:,:], F_c)

            # Assign appropriate signs and sum MFs across phase speeds.
            MF[:,lev,:] = torch.sum(c_hat0*F_c*above_source[:,lev:lev+1,:],dim=3).squeeze()
        
        if split:
            MFx, MFy = self.direction[0]*MF, self.direction[1]*MF
            MF = {"MFx":MFx,"MFy":MFy}
        return MF
    def _init_wave_packet(self, wave_packet_info:tuple) -> tuple:
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
        mom_flux = B_m*torch.special.exp2(-((phase_speeds-c_0)/c_w)**2)
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

        return torch.linspace(-c_max, c_max, int(2 * c_max / dc + 1))
