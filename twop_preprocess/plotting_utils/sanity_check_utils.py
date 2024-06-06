import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_trace(F, rois, plot_baseline=False, baseline=0, ncols=2, icol=0, linecolor='b', title="F"):
    for i, roi in enumerate(rois):
        plt.subplot2grid((len(rois), ncols), (i, icol))
        plt.plot(F[roi, :], c=linecolor)
        plt.title(f"ROI{roi} {title}")
        if plot_baseline:
            if len(baseline[roi,:])==1:
                plt.axhline(baseline[roi,:], color="r")
            else:
                plt.plot(baseline[roi, :], color="r")
                
                
def plot_raw_trace(F, random_rois, Fneu=[], titles=["F", "Fneu"]):
    plt.figure(figsize=(10, 3*len(random_rois)))
    plot_trace(F, random_rois, ncols=2, icol=0, title=titles[0])
    if len(Fneu)>0:
        plot_trace(Fneu, random_rois, ncols=2, icol=1, title=titles[1])
    plt.tight_layout()
    
    
def plot_detrended_trace(F_original, F_trend, F_detrended, Fneu_original, Fneu_trend, Fneu_detrended, random_rois):
    plt.figure(figsize=(20, 3*len(random_rois)))
    plot_trace(F_original, random_rois, plot_baseline=True, baseline=F_trend,ncols=4, icol=0, title="F")
    plot_trace(F_detrended, random_rois, ncols=4, icol=1, title="F_detrended")
    plot_trace(Fneu_original, random_rois, plot_baseline=True, baseline=Fneu_trend, ncols=4, icol=2, title="Fneu")
    plot_trace(Fneu_detrended, random_rois, ncols=4, icol=3, title="Fneu_detrended")
    plt.tight_layout()
    
    
def plot_dff(Fast, dff, F0, random_rois):
    plt.figure(figsize=(15, 3*len(random_rois)))
    plot_trace(Fast, random_rois, plot_baseline=True, baseline=F0, ncols=3, icol=0, title="Fast")
    plot_trace(dff, random_rois, ncols=3, icol=1, title="dff")
    for i, roi in enumerate(random_rois):
        plt.subplot2grid((len(random_rois), 3), (i, 2))
        plt.hist(dff[i,:],bins=50)
        plt.title(f"median {np.round(np.median(dff[i,:]),2)}, mode {np.round(stats.mode(dff[i,:])[0][0],2)}")
    plt.tight_layout()