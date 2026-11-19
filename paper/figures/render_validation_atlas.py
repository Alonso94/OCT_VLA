import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.edgecolor':'#737d83','xtick.color':'#42494e','ytick.color':'#42494e'})
fig=plt.figure(figsize=(7.15,4.45),facecolor='white')
L='#17528c'; R='#a64735'; ink='#1e2a35'; A='#dae9e9'; B='#eee1be'
def title(x,y,label,text):
 fig.text(x,y,label,weight='bold',fontsize=9,color=ink);fig.text(x+.033,y,text,fontsize=9,weight='bold',color=ink)
title(.023,.949,'A','Selection from one demonstration');title(.53,.949,'B','Push targets in the same scene')
ax=fig.add_axes([.115,.685,.335,.142])
for i,(u,v) in enumerate([('A','B'),('B','A')]):
 for x,obj in [(-15,u),(0,v)]:
  ax.add_patch(Rectangle((x,i-.42),15,.84,facecolor=A if obj=='A' else B,edgecolor='none'))
  ax.text(x+7.5,i,'Object '+obj,ha='center',va='center',fontsize=8,color=ink)
ax.axvline(0,color='#788184',lw=.8,ls=':');ax.set(xlim=(-15,15),ylim=(1.5,-.5),yticks=[0,1],yticklabels=['Nearest','Farthest'],xticks=[-15,0,15],xlabel='Marker position, x (cm)');ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
ax=fig.add_axes([.54,.625,.43,.26]);ax.set_xlim(-15,15);ax.set_ylim(-12,6);ax.set_aspect('equal');ax.axis('off')
for x,obj,fill in [(-10,'A',A),(10,'B',B)]:
 ax.add_patch(Rectangle((x-1.4,-.55),2.8,2.1,facecolor=fill,edgecolor='#455057',lw=.7));ax.text(x,.5,obj,ha='center',va='center',fontsize=7)
 ax.annotate('',xy=(x,-.75),xytext=(-1,-7.7),arrowprops=dict(arrowstyle='->',color=L,lw=1.35,linestyle='-' if x>0 else '--',connectionstyle='arc3,rad='+('-.12' if x<0 else '.12')))
ax.scatter([-1],[-7.7],s=23,c=L,zorder=5);ax.text(-1,-9.8,'Left wrist',va='center',ha='center',color=L,weight='bold')
ax.scatter([12],[-6.6],s=26,marker='X',c=R);ax.text(11.8,-9.8,'Right wrist',ha='center',va='center',color=R,weight='bold')
ax.scatter([5],[.3],s=14,marker='D',c=ink);ax.text(4,3.1,'Marker',ha='center',fontsize=7)
ax.text(-9,-3.4,'Farthest',ha='center',fontsize=6.7);ax.text(7.7,-3.4,'Nearest',ha='left',fontsize=6.7)
title(.023,.505,'C','Reactive handover');fig.text(.40,.505,'Dashed  nominal     /     Solid  delayed receiver',fontsize=7.6,color='#4b535a')
ax=fig.add_axes([.12,.135,.64,.30]);t=np.linspace(0,1.4,280)
def draw(times,vals,col,style):
 ax.plot(t,np.interp(t,times,vals),color=col,ls=style,lw=1.25 if style=='--' else 1.8)
ax.axvspan(.68,1.00,color='#e9eef2',zorder=0);ax.text(.84,0,'Extra hold',ha='center',va='center',fontsize=7,color='#343c43',bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1))
for delay,style in [(0,'--'),(.32,'-')]:
 draw([0,.36,.68+delay,1.0+delay,1.4],[-10,-2,-2,-10,-10],L,style)
 draw([0,.22,.55+delay,.72+delay,1.05+delay,1.4],[10,10,2,2,10,10],R,style)
ax.set(xlim=(0,1.4),ylim=(-12,12),yticks=[-10,0,10],xticks=[0,.35,.7,1.05,1.4],xlabel='Time from start of exchange (s)',ylabel='Wrist x (cm)')
ax.text(1.44,10,'Right wrist\nreceiver',color=R,weight='bold',va='center',fontsize=7.8);ax.text(1.44,-10,'Left wrist\ngiver',color=L,weight='bold',va='center',fontsize=7.8)
fig.text(.023,.012,'Analytic reference trajectories',fontsize=7,color='#58636b')
out=Path(__file__).resolve().parent
fig.savefig(out/'BimanualPFN_Validation_Atlas_Example.png',dpi=380)
fig.savefig(out/'BimanualPFN_Validation_Atlas.pdf')
