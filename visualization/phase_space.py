#Fonctions de rendu pour l'espace de phase 3D.
#Toutes les fonctions retournent une Figure matplotlib

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D           
from mpl_toolkits.mplot3d.art3d import Line3DCollection

DARK_BG = "#07080f"
TEXT_COLOR = "#aabbdd"

def _dark_ax3d(fig, pos = 111):
  """Crée un axe 3D avec thème sombre."""
  ax = fig.add_subplot(pos, projection="3d")
  ax.set_facecolor(DARK_BG)
  fig.patch.set_facecolor(DARK_BG)
  for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor("#1a1f35")
  for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.line.set_color("#1a1f35")
    axis.set_tick_params(labelcolor=TEXT_COLOR, labelsize=7)
  ax.set_xlabel("X", color=TEXT_COLOR, labelpad=4, fontsize=9)
  ax.set_ylabel("Y", color=TEXT_COLOR, labelpad=4, fontsize=9)
  ax.set_zlabel("Z", color=TEXT_COLOR, labelpad=4, fontsize=9)
  return ax

def plot_trajectory_3d(traj,title,colormap = "plasma",alpha = 0.85,lw = 0.5,elev = 20,azim = 45,figsize = (9, 7)):
  """
  Trace la trajectoire 3D colorée selon la vitesse instantanée.
 
  La couleur encode la vitesse ‖dstate/dt‖ normalisée.
  Les zones lentes (rouge/violet) correspondent aux "ailes"
  où la trajectoire ralentit avant de changer d'aile.
  Les zones rapides (jaune) sont les transitions entre ailes.
 
  Args
    traj      : (n_steps, 3)
    colormap  : nom d'une colormap matplotlib ('plasma', 'cool', 'viridis'…)
  """
  fig = plt.figure(figsize=figsize, facecolor=DARK_BG)
  ax = _dark_ax3d(fig)
 
  n = len(traj)
  # Vitesse normalisée → couleur
  diff  = np.diff(traj, axis=0)
  speed = np.linalg.norm(diff, axis=1)
  speed = (speed - speed.min()) / (speed.max() - speed.min() + 1e-12)
 
  cmap = cm.get_cmap(colormap)
 
  # Construction de segments pour Line3DCollection (une couleur par segment)
  # segments[i] = [[x_i, y_i, z_i], [x_{i+1}, y_{i+1}, z_{i+1}]]
  pts      = traj.reshape(-1, 1, 3)
  segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
  colors   = cmap(speed)
  colors[:, 3] = alpha  # canal alpha
 
  lc = Line3DCollection(segments, colors=colors, linewidths=lw)
  ax.add_collection3d(lc)
 
  # Point courant (dernier point)
  ax.scatter(*traj[-1], color="white", s=18, zorder=5)
 
  # Ajustement des axes
  margin = 3
  ax.set_xlim(traj[:, 0].min() - margin, traj[:, 0].max() + margin)
  ax.set_ylim(traj[:, 1].min() - margin, traj[:, 1].max() + margin)
  ax.set_zlim(traj[:, 2].min() - margin, traj[:, 2].max() + margin)
 
  ax.set_title(title, color="white", fontsize=11, pad=10)
  ax.view_init(elev=elev, azim=azim)
 
  # Barre de couleur
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
  sm.set_array([])
  cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.04, aspect=20)
  cbar.set_label("Vitesse normalisée", color=TEXT_COLOR, fontsize=8)
  cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)
 
  plt.tight_layout()
  return fig



def plot_ensemble_3d(ensemble_traj,title= "Dispersion de l'ensemble",figsize = (9, 7)):
  """
  Trace N trajectoires d'un ensemble (visualisation de la sensibilité aux CI).
 
  ensemble_traj : (n_steps, N, 3)
  Chaque trajectoire a une couleur différente, toutes transparentes.
  """
  fig = plt.figure(figsize=figsize, facecolor=DARK_BG)
  ax = _dark_ax3d(fig)
 
  n_traj = ensemble_traj.shape[1]
  colors = cm.cool(np.linspace(0, 1, n_traj))
 
  for j in range(n_traj):
    traj_j = ensemble_traj[:, j, :]
    ax.plot(traj_j[:, 0], traj_j[:, 1], traj_j[:, 2],color=colors[j], lw=0.4, alpha=0.6,)
 
  ax.set_title(title, color="white", fontsize=11, pad=10)
 
  all_pts = ensemble_traj.reshape(-1, 3)
  margin = 3
  ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
  ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
  ax.set_zlim(all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin)
 
  plt.tight_layout()
  return fig
 