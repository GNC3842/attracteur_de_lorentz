"""
visualization/diagnostics.py
=============================
Graphes diagnostiques : séries temporelles, divergence,
exposant de Lyapunov, carte de retour.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DARK_BG = "#07080f"
GRID_COLOR = "#1a1f35"
TEXT_COLOR = "#aabbdd"
COLORS = {"x": "#00e5ff", "y": "#ff6e40", "z": "#69ff47"}


def _style_ax(ax):
  """Applique le thème sombre à un axe 2D."""
  ax.set_facecolor(DARK_BG)
  ax.tick_params(colors=TEXT_COLOR, labelsize=8)
  for spine in ax.spines.values():
    spine.set_color(GRID_COLOR)
  ax.grid(color=GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.6)
  ax.xaxis.label.set_color(TEXT_COLOR)
  ax.yaxis.label.set_color(TEXT_COLOR)
  ax.title.set_color("white")


def plot_time_series(traj,dt,figsize = (11, 5)):
  """
  Trace x(t), y(t), z(t) sur trois sous-graphes empilés.
  Permet de voir les oscillations irrégulières caractéristiques du chaos.
  """
  n = len(traj)
  t = np.arange(n) * dt

  fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, facecolor=DARK_BG)
  fig.suptitle("Séries temporelles", color="white", fontsize=11)

  labels = ["x(t)", "y(t)", "z(t)"]
  for i, (ax, label, color) in enumerate(zip(axes, labels, COLORS.values())):
    ax.plot(t, traj[:, i], color=color, lw=0.7, alpha=0.9)
    ax.set_ylabel(label, fontsize=9)
    _style_ax(ax)

  axes[-1].set_xlabel("Temps t", fontsize=9)
  plt.tight_layout()
  return fig


def plot_divergence(traj1,traj2,dt,delta0= 1e-8,figsize = (10, 4)):
  """
  Trace la divergence entre deux trajectoires aux CI quasi-identiques.

  Deux panneaux :
    - Gauche : x₁(t) vs x₂(t) — on voit à l'œil quand elles divergent
    - Droite : log(distance) vs t — doit être linéaire si chaos (pente = λ)
  """
  n = min(len(traj1), len(traj2))
  t = np.arange(n) * dt

  dist = np.linalg.norm(traj1[:n] - traj2[:n], axis=1)
  dist = np.maximum(dist, 1e-15)   # éviter log(0)

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor=DARK_BG)
  fig.suptitle("Sensibilité aux conditions initiales", color="white", fontsize=11)

  # Panneau gauche : les deux x(t)
  ax1.plot(t, traj1[:n, 0], color=COLORS["x"], lw=0.8, label="CI #1", alpha=0.9)
  ax1.plot(t, traj2[:n, 0], color=COLORS["y"], lw=0.8, label="CI #2", alpha=0.8)
  ax1.set_xlabel("t", fontsize=9)
  ax1.set_ylabel("x(t)", fontsize=9)
  ax1.set_title(f"Écart initial : {delta0:.0e}", fontsize=9)
  ax1.legend(fontsize=8, facecolor="#0d1220", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
  _style_ax(ax1)

  # Panneau droit : log de la distance
  log_dist = np.log10(dist)
  ax2.plot(t, log_dist, color="#ffcc00", lw=0.8)
  ax2.axhline(0, color=GRID_COLOR, lw=0.5, linestyle="--")
  ax2.set_xlabel("t", fontsize=9)
  ax2.set_ylabel("log₁₀(distance)", fontsize=9)
  ax2.set_title("Divergence exponentielle", fontsize=9)
  _style_ax(ax2)

  plt.tight_layout()
  return fig


def plot_lyapunov(exponents,dt,figsize = (9, 4)):
  """
  Trace la convergence de l'exposant de Lyapunov au cours du temps.

  La courbe doit se stabiliser autour de λ ≈ 0.9 pour Lorenz classique.
  La phase de convergence initiale (bruyante) dure ~100-500 unités de temps.
  """
  n = len(exponents)
  t = np.arange(1, n + 1) * dt

  final_val = exponents[-1]

  fig, ax = plt.subplots(figsize=figsize, facecolor=DARK_BG)
  ax.plot(t, exponents, color="#00e5ff", lw=0.9, label="λ(t)")
  ax.axhline(final_val, color="#ff6e40", lw=1, linestyle="--",
               label=f"Valeur convergée : {final_val:.4f}")
  ax.axhline(0, color=GRID_COLOR, lw=0.5, linestyle=":")

  ax.set_xlabel("Temps t", fontsize=9)
  ax.set_ylabel("Exposant de Lyapunov λ", fontsize=9)
  ax.set_title("Convergence de l'exposant de Lyapunov maximal", fontsize=10)
  ax.legend(fontsize=9, facecolor="#0d1220", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
  _style_ax(ax)

  # Annotation de la valeur finale
  ax.annotate(
      f"λ = {final_val:.3f}  →  {'chaos' if final_val > 0 else 'stable'}",
      xy=(t[-1] * 0.6, final_val),
      color="#ff6e40", fontsize=9,
  )

  plt.tight_layout()
  return fig

