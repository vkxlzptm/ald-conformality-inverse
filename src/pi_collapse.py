"""Which dimensionless group collapses the profile shape: AR*s0 or AR*sqrt(s0)?"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from trench_mc import profile, NBIN
zc=(np.arange(NBIN)+0.5)/NBIN

def shape(AR,s0,dose_mult=0.15):
    spb=400.0*AR
    dose=int(dose_mult*spb*NBIN)          # same coverage level (Pi2 matched)
    th,_,_,_=profile(AR,dose,s0=s0,sites_per_bin=spb,seed=7)
    th=th[:-1]
    return th/max(th[0],1e-12)

fig,axes=plt.subplots(1,2,figsize=(12.4,5.2)); fig.subplots_adjust(wspace=0.28)
famA=[(10,0.040),(20,0.020),(40,0.010)]           # AR*s0 = 0.4
famB=[(10,0.040),(20,0.010),(40,0.0025)]          # AR*sqrt(s0) = 2.0
for ax,fam,name,expr in [(axes[0],famA,r"$\Pi=AR\cdot s_0$ = 0.4","AR*s0"),
                          (axes[1],famB,r"$\Pi=AR\sqrt{s_0}$ = 2.0","AR*sqrt(s0)")]:
    for (AR,s0),c in zip(fam,["#2874a6","#1e8449","#c0392b"]):
        y=shape(AR,s0)
        ax.plot(y,zc[:-1],lw=2.2,color=c,label=f"AR={AR}, $s_0$={s0}")
        print(name,AR,s0,"bottom/top=%.3f"%y[-1])
    ax.set_xlabel(r"Normalized coverage  $\theta(z)/\theta(0)$",fontsize=12)
    ax.set_ylabel("Normalized depth  z / H",fontsize=12)
    ax.invert_yaxis(); ax.set_xlim(0,1.05); ax.grid(alpha=0.3); ax.legend(fontsize=10.5)
    ax.set_title(name,fontsize=13)
fig.suptitle("Profile-shape collapse test (low dose, linear regime)",fontsize=14.5,y=1.01)
fig.savefig("pi_collapse.png",dpi=155,bbox_inches="tight",facecolor="white")
print("saved")
