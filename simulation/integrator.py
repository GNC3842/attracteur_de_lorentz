#Intégrateur numérique pour equa diff 1D
#Utilisation de la méthode de Runge-Kutta RK4 plutot que EUler 
#Pourquoi ? L'erreur en RK4 est en O(dt⁴) alors que EUler est en O(dt) ie RK4 plus précise mais demande plus de calculs
#La précision est de mise car systeme chaotique


from __future__ import annotations
import numpy as np
from simulation.system import ChaoticSysteme

######
# Définition du pas élémentaire en RK4
######

def rk4_step(system,state,dt):
  """
  Calcul un pas élémentaire en RK4:
      - On évlaue la pente au début: k1
      - On évalue la pente au milieu du pas estimé avec k1: k2
      - On évalue la pente au milieu du pas estimé avec k2: k3 
      - On évalue la pente à la fin du pas estimé avec k3: k4
    On obtient ainsi une "combinaison pondéré" (1/6, 2/6, 2/6, 1/6)
  Args:
      system (ChaoticSyteme): le système que l'on va évalué
      state (ndarray): l'état 
      dt (float): le pas de temps
  """
  k1 = system.deriv(state)
  k2 = system.deriv(state + 0.5*dt*k1)
  k3 = system.deriv(state + 0.5*dt*k2)
  k4 = system.deriv(state + dt*k3)

  return state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

##########
# Intégration de la trajectorie
##########

def integrate(system,state0,n_steps,dt):
  """
  Integre une trajectorie sure n_steps pas de temps 
  Args:
      system (ChaoticSysteme): le systeme étudié
      state0 (3,): les CI
      n_steps (int): nombre de pas de temps
      dt (float): pas de temps
  Return:
      trajectoire - tableau ndarray (n_steps, 3) où trajectorie[i] = état au temps i*dt
  """

  traj = np.empty((n_steps,3))
  traj[0] = state0
  for i in range(1,n_steps):
    traj[i] = rk4_step(system=system,state=traj[i-1],dt=dt)
  return traj


########
# Intégrer plusieurs trajectoires d'un coup
########
def multi_integrate(system,states0,n_steps,dt):
  """
  Cette fois state0 est de la forme (N,3) où N est donc le nombre de CI différentes
  ET 
  trajectoires - tableau ndarray (n_steps,N, 3) où trajectorie[i][j] = état de la trajectoire j au temps i*dt
  """
  n = states0.shape[0]
  traj = np.empty((n_steps,n,3))
  traj[0] = states0

  for i in range(1,n_steps):
    traj[i] = rk4_step(system,traj[i-1],dt)
  return traj


###
# Calcul de la vitesse intantanée
###
def compute_speed(traj):
  """    
  Calcule la vitesse normalisée en chaque point de la trajectoire.
  Return:
       array (n_steps - 1,) dans [0, 1]
  """
  diff = np.diff(traj, axis=0)
  speed = np.linalg.norm(diff, axis=1)
  vmin, vmax = speed.min(), speed.max()
  if vmax - vmin < 1e-12:
    return np.zeros_like(speed)
  return (speed - vmin) / (vmax - vmin)