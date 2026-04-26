"""
visualization/animation.py
==========================
Animations en temps réel du système de Lorenz.

Deux fonctions principales :
  - animate_trajectory() : trajectoire 3D qui se dessine en direct
  - animate_divergence()  : trajectoire 3D + divergence chaotique côte à côte

Principe :
    FuncAnimation appelle update() à intervalle régulier.
    À chaque frame, on intègre STEPS_PER_FRAME nouveaux points,
    on les ajoute au buffer, et on met à jour les objets graphiques
    sans recréer la figure — ce qui donne l'animation fluide.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection

from simulation.system     import ChaoticSysteme, LorenzSystem
from simulation.integrator import rk4_step, compute_speed


# Thème
BG = "#07080f"
BG2 = "#0d1220"
TEXT = "#aabbdd"
GRID = "#1a1f35"


### FONCTIONS HELPERS INTERNES

def _style_ax3d(ax, fig):
  """Applique le thème sombre à un axe 3D."""
  ax.set_facecolor(BG)
  fig.patch.set_facecolor(BG)
  for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor(GRID)
  for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.line.set_color(GRID)
    axis.set_tick_params(labelcolor=TEXT, labelsize=7)
  ax.set_xlabel("X", color=TEXT, labelpad=4, fontsize=9)
  ax.set_ylabel("Y", color=TEXT, labelpad=4, fontsize=9)
  ax.set_zlabel("Z", color=TEXT, labelpad=4, fontsize=9)


def _style_ax2d(ax):
  """Applique le thème sombre à un axe 2D."""
  ax.set_facecolor(BG2)
  ax.tick_params(colors=TEXT, labelsize=8)
  for spine in ax.spines.values():
    spine.set_color(GRID)
  ax.grid(color=GRID, linewidth=0.4, linestyle="--", alpha=0.6)
  ax.xaxis.label.set_color(TEXT)
  ax.yaxis.label.set_color(TEXT)
  ax.title.set_color("white")


def _colored_segments(pts, cmap):
  """
  Construit une Line3DCollection colorée selon la position dans la traîne.
  pts : (n, 3)
  Retourne la collection prête à être ajoutée à un axe 3D.
  """
  n = len(pts)
  if n < 2:
    return Line3DCollection([], linewidths=0.8)

  segs = np.stack([pts[:-1], pts[1:]], axis=1)       # (n-1, 2, 3)
  ages = np.linspace(0, 1, n - 1)                    # 0 = vieux, 1 = récent
  colors = cmap(ages)
  colors[:, 3] = ages * 0.9 + 0.05                     # alpha croissant

  lc = Line3DCollection(segs, colors=colors, linewidths=0.8)
  return lc


# I - Trajectorei 3D

def animate_trajectory(system = LorenzSystem() ,dt = 0.006,tail = 1500,steps_per_frame = 6,max_frames = 3000,colormap = "plasma",auto_rotate = True, figsize = (9, 7)) :
  """
  Anime la trajectoire 3D en temps réel.
  Args
    system          : instance de ChaoticSystem (défaut : LorenzSystem())
    dt              : pas de temps RK4
    tail            : nombre de points affichés simultanément (traîne)
    steps_per_frame : points calculés par frame (contrôle la vitesse)
    max_frames      : nombre de frames avant arrêt (None = infini)
    colormap        : colormap matplotlib pour la traîne
    auto_rotate     : rotation automatique de la caméra
  """
  if system is None:
    system = LorenzSystem()

  cmap  = plt.get_cmap(colormap)
  state = system.default_state.copy().astype(float)
  buf   = [state.copy()]          # buffer de points
  t_val = [0.0]

  # Figure
  fig = plt.figure(figsize=figsize, facecolor=BG)
  fig.suptitle(f"Attracteur de {system.name},"f"[dt={dt}  traîne={tail}]", color="white", fontsize=11)
  ax = fig.add_subplot(111, projection="3d")
  _style_ax3d(ax, fig)

  # Limites fixes basées sur une pré-intégration courte
  _pre = system.default_state.copy().astype(float)
  _pts = np.array([_pre := rk4_step(system, _pre, dt) for _ in range(3000)])
  margin = 4
  ax.set_xlim(_pts[:, 0].min() - margin, _pts[:, 0].max() + margin)
  ax.set_ylim(_pts[:, 1].min() - margin, _pts[:, 1].max() + margin)
  ax.set_zlim(_pts[:, 2].min() - margin, _pts[:, 2].max() + margin)

  # Objets graphiques — initialisés vides, mis à jour dans update()
  lc_holder  = [ax.add_collection3d(_colored_segments(np.zeros((2, 3)), cmap))]
  point_plot, = ax.plot([], [], [], "o", color="white", markersize=5, zorder=5)
  time_text   = ax.text2D(0.02, 0.97, "", transform=ax.transAxes,color=TEXT, fontsize=8, va="top")

  azim = [45.0]

  def update(frame):
    nonlocal state
    # Intégration de steps_per_frame nouveaux points
    for _ in range(steps_per_frame):
      state = rk4_step(system, state, dt)
      buf.append(state.copy())
      t_val[0] += dt

    # Garder uniquement les `tail` derniers points
    if len(buf) > tail:
      del buf[: len(buf) - tail]

    pts = np.array(buf)

    # Mise à jour de la collection colorée
    lc_holder[0].remove()
    lc_new = _colored_segments(pts, cmap)
    ax.add_collection3d(lc_new)
    lc_holder[0] = lc_new

    # Point courant
    point_plot.set_data([pts[-1, 0]], [pts[-1, 1]])
    point_plot.set_3d_properties([pts[-1, 2]])

    #Rotation automatique
    if auto_rotate:
      azim[0] = (azim[0] + 0.25) % 360
      ax.view_init(elev=20, azim=azim[0])

    # Texte 
    time_text.set_text(f"t = {t_val[0]:.2f}   pts = {len(buf)}")

    return lc_holder[0], point_plot, time_text

  ani = animation.FuncAnimation(fig, update,frames=max_frames,interval=20,blit=False,repeat=False)

  plt.tight_layout()
  plt.show()


# ─────────────────────────────────────────────────────────────
#  2. Trajectoire 3D + divergence chaotique côte à côte
# ─────────────────────────────────────────────────────────────

def animate_divergence(system = LorenzSystem(),delta0 = 1e-8,dt = 0.006,tail = 1500,steps_per_frame = 6,max_frames: int = 3000,colormap = "cool",figsize = (14, 6)):
  """
  Anime côte à côte :
    - Gauche  : trajectoire 3D principale (traîne colorée)
    - Droite  : x₁(t) et x₂(t) superposés, révélant la divergence chaotique

  delta0 : écart initial entre les deux trajectoires (typiquement 1e-8)
  """

  cmap = plt.get_cmap(colormap)

  # Deux états initiaux : perturbation infinitésimale sur x
  state1 = system.default_state.copy().astype(float)
  state2 = state1.copy()
  state2[0] += delta0

  buf1 = [state1.copy()]
  t_arr = [0.0]
  x1_arr = [state1[0]]
  x2_arr = [state2[0]]

  # Figure 
  fig = plt.figure(figsize=figsize, facecolor=BG)
  fig.suptitle(f"Divergence chaotique — {system.name}   δ₀ = {delta0:.0e}",color="white", fontsize=12)

  ax3d = fig.add_subplot(121, projection="3d")
  ax2d = fig.add_subplot(122)
  _style_ax3d(ax3d, fig)
  _style_ax2d(ax2d)

  # Pré-intégration pour fixer les limites 3D
  _pre = system.default_state.copy().astype(float)
  _pts = np.array([_pre := rk4_step(system, _pre, dt) for _ in range(3000)])
  margin = 4
  ax3d.set_xlim(_pts[:, 0].min() - margin, _pts[:, 0].max() + margin)
  ax3d.set_ylim(_pts[:, 1].min() - margin, _pts[:, 1].max() + margin)
  ax3d.set_zlim(_pts[:, 2].min() - margin, _pts[:, 2].max() + margin)
  ax3d.set_title("Espace de phase 3D", color="white", fontsize=10, pad=8)

  # Objets 3D
  lc_holder  = [ax3d.add_collection3d(_colored_segments(np.zeros((2, 3)), cmap))]
  point_plot, = ax3d.plot([], [], [], "o", color="white", markersize=5, zorder=5)

  # Objets 2D
  line1, = ax2d.plot([], [], color="#00e5ff", lw=0.9, label="CI #1", alpha=0.95)
  line2, = ax2d.plot([], [], color="#ff6e40", lw=0.9, label="CI #2", alpha=0.85)
  ax2d.set_xlabel("t", fontsize=9)
  ax2d.set_ylabel("x(t)", fontsize=9)
  ax2d.set_title(f"x(t) — deux trajectoires (δ₀ = {delta0:.0e})", color="white", fontsize=10)
  ax2d.legend(fontsize=8, facecolor="#0d1220", edgecolor=GRID, labelcolor=TEXT)

  # Annotation "divergence" — cachée au début
  div_annot = ax2d.annotate("↑ divergence", xy=(0, 0), fontsize=9, color="#ff6e40", alpha=0)

  time_text = ax3d.text2D(0.02, 0.97, "", transform=ax3d.transAxes,color=TEXT, fontsize=8, va="top")
  azim = [45.0]
  diverged = [False]

  def update(frame):
    nonlocal state1, state2

    #Intégration 
    for _ in range(steps_per_frame):
      state1 = rk4_step(system, state1, dt)
      state2 = rk4_step(system, state2, dt)
      t_now = t_arr[-1] + dt
      t_arr.append(t_now)
      x1_arr.append(state1[0])
      x2_arr.append(state2[0])

    buf1.append(state1.copy())
    if len(buf1) > tail:
      del buf1[: len(buf1) - tail]

    pts = np.array(buf1)
    t = np.array(t_arr)
    x1 = np.array(x1_arr)
    x2 = np.array(x2_arr)

    #Mise à jour 3D 
    lc_holder[0].remove()
    lc_new = _colored_segments(pts, cmap)
    ax3d.add_collection3d(lc_new)
    lc_holder[0] = lc_new

    point_plot.set_data([pts[-1, 0]], [pts[-1, 1]])
    point_plot.set_3d_properties([pts[-1, 2]])

    azim[0] = (azim[0] + 0.2) % 360
    ax3d.view_init(elev=20, azim=azim[0])
    time_text.set_text(f"t = {t[-1]:.2f}")

    #Mise à jour 2D 
    line1.set_data(t, x1)
    line2.set_data(t, x2)

    ax2d.set_xlim(0, max(t[-1], 1))
    ymin = min(x1.min(), x2.min()) - 2
    ymax = max(x1.max(), x2.max()) + 2
    ax2d.set_ylim(ymin, ymax)

    # Détecter la divergence : écart > 1 unité
    gap = abs(x1[-1] - x2[-1])
    if gap > 1.0 and not diverged[0]:
      diverged[0] = True
      div_annot.set_position((t[-1], x2[-1]))
      div_annot.set_alpha(1.0)

    return lc_holder[0], point_plot, line1, line2, time_text

    ani = animation.FuncAnimation(
        fig, update,
        frames=max_frames,
        interval=20,
        blit=False,
        repeat=False,
    )

    plt.tight_layout()
    plt.show()