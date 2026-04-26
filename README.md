# Attracteur de Lorenz — Simulation interactive

Simulation numérique et analyse du système de Lorenz.  
Inclut la visualisation 3D, le calcul de l'exposant de Lyapunov, la section de Poincaré et la dispersion d'ensemble.

---

## Prérequis

- Python 3.10+

---

## Installation

```bash
# Cloner / télécharger le projet
cd lorenz_project

# Créer un environnement virtuel (recommandé)
python -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Lancer le programme
python3 app.py
```
---

## Structure du projet

```
lorenz_project/
│
├── simulation/
│   ├── system.py          — Classes LorenzSystem et RosslerSystem
│   └── integrator.py      — RK4 scalaire et vectorisé (ensemble)
│
├── analysis/
│   ├── lyapunov.py        — Exposant de Lyapunov (algorithme de Benettin)
│   └── poincare.py        — Section de Poincaré et carte de retour
│
├── visualization/
│   ├── phase_space.py     — Rendu 3D (trajectoire + ensemble)
│   └── diagnostics.py     — Séries temporelles, divergence, Lyapunov, Poincaré
│
├── requirements.txt
└── README.md
```

---

## Fonctionnalités

| Section | Description | Paramètres ajustables |
|---------|-------------|----------------------|
| Phase 3D | Trajectoire colorée par vitesse, rotation libre | σ, ρ, β, colormap, vue |
| Lyapunov | Courbe de convergence + divergence des CI | σ, ρ, nb de pas, δ₀ |
| Poincaré | Section z = ρ−1 + carte de retour xₙ → xₙ₊₁ | σ, ρ, nb de points, direction |
| Ensemble | N trajectoires depuis une boule de rayon ε | N, ε, nb de pas |
| Séries | x(t), y(t), z(t) empilées | — |

---

## Paramètres du système

Les équations de Lorenz :

```
dx/dt = σ(y − x)
dy/dt = x(ρ − z) − y
dz/dt = xy − βz
```

| Paramètre | Valeur classique | Effet |
|-----------|-----------------|-------|
| σ (sigma) | 10.0 | Nombre de Prandtl — couplage convection/diffusion |
| ρ (rho)   | 28.0 | Rayleigh normalisé — **chaos si ρ > 24.74** |
| β (beta)  | 8/3 ≈ 2.667 | Facteur géométrique — amortissement thermique |

Valeurs de référence : exposant de Lyapunov λ ≈ 0.906, dimension fractale ≈ 2.06.

---

## Dépendances

```
numpy        — calcul numérique vectorisé
matplotlib   — graphes 2D et 3D
ipywidgets   — sliders et boutons dans Jupyter
ipympl       — backend matplotlib interactif (%matplotlib widget)
scipy        — optionnel, pour RK45 adaptatif
```
