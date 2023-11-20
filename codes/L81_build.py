#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.append('../codes/')
import intake
import xarray as xr
import numpy as np
from ngc3ICON import ngc3ICON as n3
from L81 import Lindzen1981 as L81
from utils import *
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import torch


# In[3]:


zoom=6
basedir="/work/bm1233/icon_for_ml/spherical/nextgems3/nofilter/"
dsYx = xr.open_zarr(f'{basedir}res102km_MFx.zarr/')
dsYy = xr.open_zarr(f'{basedir}res102km_MFy.zarr/')
cat = intake.open_catalog("/work/bm1235/k203123/NextGEMS_Cycle3.git/experiments/ngc3028/outdata/ngc3028.yaml")
dsX = cat.ngc3028(zoom=zoom,time="PT3H",chunks={"cell": 12 * 4**zoom, "time": 1, "level_full": 90, "level_half": 91},).to_dask()


# ### Variables That appear Explicitly
# - u 
# - v 
# - k = 100_000 m 
# - N (need temperature, heights)
# - density (need pressure, temperature)
# - c phase speeds 
# 
# 
# ### Variables we need from the dataset to compute the above ones.
# - U "ua"
# - V "va"
# - T "ta"
# - Z "zg"
# - P "phalf" or "pfull" depending on use.
# 

# In[4]:


npix=dsX.cell[-1].data+1
masked_ind = list(ang_range(npix, 25, 50, 155, 215))
mask = np.zeros(npix,dtype=bool)
mask[masked_ind] = True
time_range=["2021-12-1","2021-12-1"]


# In[5]:


worldmap(mask)


# In[6]:


data = n3(dsX, dsYx, dsYy, mask, time_range, source_lev=200e2)


# In[7]:


direction="SE"
direction = np.pi/6
l81=L81(direction, B_m=1.5e-3)
levs=torch.Tensor(range(90)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
above_source = levs <= data.source_levs.unsqueeze(1)


# In[8]:


rho0=torch.gather(data.rho, 1, data.source_levs.unsqueeze(1))
wind = (l81.direction[0] * data.u) + (l81.direction[1] * data.v)
wind0=torch.gather(wind, 1, data.source_levs.unsqueeze(1))
F_c = (rho0*l81.mom_flux)
c_hat0 = torch.sign(l81.phase_speeds-wind0)
c_hat = torch.sign(l81.phase_speeds-wind)
breaking_cond = data.rho * l81.h_wavenum*torch.abs(l81.phase_speeds - wind)**3 / (2*data.N)
breaking_cond = torch.where(c_hat == c_hat0, breaking_cond,0)


# In[9]:


pred_MF = torch.zeros(data.MFx.shape)
F_c000 = torch.zeros(l81.phase_speeds.shape[0],91)
F_c000[:,-1] = F_c[0,0,0,:]
for lev in range(data.nlev-1,-1,-1):
    #lev=69
    breaking= F_c >= breaking_cond[:,lev:lev+1,:,:]
    update = breaking * above_source[:,lev:lev+1,:,:]
    F_c = torch.where(update, breaking_cond[:,lev:lev+1,:,:], F_c)
    F_c000[:,lev] = F_c[0,0,0,:]
    pred_MF[:,lev,:] = torch.sum(c_hat0*F_c*above_source[:,lev:lev+1,:], dim=3).squeeze()


# In[10]:


plt.plot(l81.phase_speeds, c_hat[0,0,0,:]*F_c000[:,-1], linewidth=10, label="initial")
plt.plot(l81.phase_speeds, c_hat[0,0,0,:]*F_c000[:,69], linewidth=5, label="source level")
for d in [20,40,60,69]:
    plt.plot(l81.phase_speeds, c_hat[0,0,0,:]*F_c000[:,69-d], label=f"{d} updates after source level")
plt.legend()
# plt.plot(l81.phase_speeds, np.sign(F_c[0,0,0,:]),label="OG momflux")
# plt.plot(l81.phase_speeds, np.sign(breaking_cond[0,0,0,:]), label="breaking conditions")
# plt.legend()


# In[ ]:





# In[14]:


t=0;c=4
pfull=dsX.pfull.isel(cell=mask, time=data.timeidx)[t,:,c]
fig,ax = plt.subplots(ncols=2,sharey=True)
MF=[data.MFx, data.MFy]
title=["Zonal MF", "Meridional MF"]
amax=max(torch.abs(MF[0][t,:,c]).max(),torch.abs(MF[1][t,:,c]).max(),torch.abs(l81.direction[0]*pred_MF[t,:,c]).max(),torch.abs(l81.direction[1]*pred_MF[t,:,c]).max())
for i in range(2):
    ax[i].plot(l81.direction[i]*pred_MF[t,:,c],pfull, label="L81 prediction")
    ax[i].plot(MF[i][t,:,c],pfull, label="target")
    ax[i].axhline(pfull[data.source_levs[t,c]],c='r',label='source level (200 hPa)')
    ax[i].set_yscale('log')
    ax[i].set_xlabel('Momentum Flux (Pa)')
    ax[i].set_title(title[i])
    ax[i].set_xlim([-amax,amax])
ax[0].set_ylabel('Pa')
ax[0].invert_yaxis()
ax[1].legend(bbox_to_anchor=(1.1, 1.05))


# In[12]:


t=0;c=0
pfull=dsX.pfull.isel(cell=mask, time=data.timeidx)[t,:,c]
fig,ax = plt.subplots()
#ax.plot(pred_MFx[t,:,c],pfull, label="L81 prediction")
ax.plot(data.MFx[t,:,c],pfull, label="target")
ax.plot(data.MFx[t,:,c+1],pfull, label="target")
ax.plot(data.MFx[t,:,c+2],pfull, label="target")

ax.axhline(pfull[data.source_levs[t,c]],c='r',label='source level (600 hPa)')

ax.legend()
ax.set_yscale('log')
ax.set_ylabel('Pa')
ax.set_xlabel('Momentum Flux (Pa)')
plt.gca().invert_yaxis()


# In[ ]:





# In[ ]:




