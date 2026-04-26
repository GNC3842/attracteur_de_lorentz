"""
analysis/poincare.py
====================
Calcul de la section de Poincaré.

Principe
--------
On choisit un plan de coupe (hyperplan) dans l'espace de phase.
Chaque fois que la trajectoire TRAVERSE ce plan dans un sens donné,
on enregistre le point d'intersection.

L'ensemble de ces points révèle la structure sous-jacente de l'attracteur :
  - Pour Lorenz, on obtient une courbe avec une structure fractale fine.
  - La dimension de cette courbe est > 1 mais < 2 → attracteur étrange.

Pourquoi l'interpolation est nécessaire
----------------------------------------
La trajectoire numérique est discrète : les points "sautent" par-dessus
le plan de coupe sans jamais l'atteindre exactement.
Si on enregistre simplement le point le plus proche, on introduit une
erreur systématique égale à dt × vitesse ≈ 0.005 × 10 = 0.05 unités.
Sur un objet fractal, cette erreur brouille complètement la structure.

Solution : interpolation linéaire entre le pas avant et le pas après
le franchissement. L'erreur descend à O(dt²) — suffisant pour voir
la structure.
"""

from __future__ import annotations
import numpy as np


def poincare_section(
    traj: np.ndarray,
    z_plane: float,
    direction: str = "up",
) -> np.ndarray:
    """
    Calcule la section de Poincaré sur le plan z = z_plane.

    Paramètres
    ----------
    traj      : trajectoire (n_steps, 3)
    z_plane   : altitude du plan de coupe (pour Lorenz : rho - 1 ≈ 27)
    direction : "up"   → traversées z_prev < z_plane ≤ z_curr
                "down" → traversées z_prev ≥ z_plane > z_curr
                "both" → les deux sens

    Retour
    ------
    points : np.ndarray (n_crossings, 2)
        Coordonnées (x, y) des intersections avec le plan.
        Peut être vide si la trajectoire ne traverse pas le plan.
    """
    points = []

    z = traj[:, 2]
    x = traj[:, 0]
    y = traj[:, 1]

    for i in range(1, len(traj)):
        z_prev, z_curr = z[i - 1], z[i]

        crossing = False
        if direction in ("up", "both") and z_prev < z_plane <= z_curr:
            crossing = True
        if direction in ("down", "both") and z_prev >= z_plane > z_curr:
            crossing = True

        if crossing:
            # Interpolation linéaire pour trouver le point exact de franchissement
            # alpha ∈ [0, 1] : fraction du pas où z = z_plane
            alpha = (z_plane - z_prev) / (z_curr - z_prev)

            x_cross = x[i - 1] + alpha * (x[i] - x[i - 1])
            y_cross = y[i - 1] + alpha * (y[i] - y[i - 1])
            points.append([x_cross, y_cross])

    if not points:
        return np.empty((0, 2))

    return np.array(points)


def poincare_return_map(
    traj: np.ndarray,
    z_plane: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construit la carte de retour (return map) de Poincaré.

    Pour chaque franchissement successif n et n+1, on trace
    x_n en fonction de x_{n+1}.  Pour un système chaotique,
    cette carte est une courbe lisse (réduction de dimension) ;
    pour un système périodique, c'est un ensemble discret de points.

    Retour
    ------
    (xn, xn1) : deux arrays de même longueur
        xn[i]  = x au franchissement i
        xn1[i] = x au franchissement i+1
    """
    pts = poincare_section(traj, z_plane, direction="up")
    if len(pts) < 2:
        return np.array([]), np.array([])

    xn  = pts[:-1, 0]
    xn1 = pts[1:,  0]
    return xn, xn1