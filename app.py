import numpy as np
import matplotlib.pyplot as plt

# Import des composants de simulation
from simulation.system import LorenzSystem
from simulation.integrator import integrate, rk4_step

# Import des outils d'analyse et de visualisation
from visualization.phase_space import plot_trajectory_3d
from visualization.diagnostics import plot_time_series, plot_divergence
from visualization.animation import animate_trajectory, animate_divergence

def main():
    # --- 1. CONFIGURATION DU SYSTÈME ---
    # Utilisation de la classe LorenzSystem avec les paramètres classiques
    lorenz = LorenzSystem(sigma=10.0, rho=28.0, beta=8/3)
    dt = 0.01
    n_pas = 8000
    state0 = lorenz.default_state # Utilise ta nouvelle méthode sans @property

    print(f"--- Simulation du système de {lorenz.name} ---")
    
    # --- 2. GÉNÉRATION DES DONNÉES ---
    # Intégration de la trajectoire principale avec RK4
    traj = integrate(lorenz, state0, n_pas, dt)
    
    # Génération d'une deuxième trajectoire pour l'étude de la divergence
    delta0 = 1e-8
    state0_perturbe = state0.copy()
    state0_perturbe[0] += delta0
    traj_perturbee = integrate(lorenz, state0_perturbe, n_pas, dt)

    # --- 3. DIAGNOSTICS STATIQUES ---
    print("Génération des graphiques de diagnostic...")
    
    # Séries temporelles x(t), y(t), z(t)
    plot_time_series(traj, dt)
    
    # Trajectoire 3D colorée par la vitesse
    plot_trajectory_3d(traj, title="Attracteur Étrange de Lorenz")
    
    # Étude de la sensibilité aux conditions initiales
    plot_divergence(traj, traj_perturbee, dt, delta0=delta0)
    
    plt.show() # Affiche tous les graphes statiques

    # --- 4. ANIMATIONS TEMPS RÉEL ---
    print("\n--- Lancement des animations ---")
    print("Note : Fermez la fenêtre de l'animation pour passer à la suivante.")
    
    # Animation de la construction de l'attracteur
    animate_trajectory(system=lorenz, dt=0.006, tail=1200)
    
    # Animation de la divergence chaotique côte à côte
    animate_divergence(system=lorenz, delta0=1e-5, dt=0.008)

if __name__ == "__main__":
    main()