#Definition des systèmes dynamiques chaotiques


from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

################
#  Classe de base abstraite d'un attracteur
################
class ChaoticSysteme:
  """
  Interface abstraite.  Tout attracteur doit implémenter :
    - deriv(state)       → vecteur dérivée
    - default_state      → conditions initiales canoniques
    - name               → nom lisible
  """
  
  name: str = "AbstractSystem"

  def deriv(self,state):
    raise NotImplementedError
  
  @property
  def default_state(self):
    raise NotImplementedError


################
# Attracteur de Lorentz
################
class LorenzSystem(ChaoticSysteme):
  """
  Régi par 3 équations:
      dx/dt = σ(y - x)
      dy/dt = x(ρ - z) - y
      dz/dt = xy - βz
  
  Ces 3 grandeurs physiques sont:
      x -> intensité de convection
      y -> différence de température ascendant/descdenat
      z -> distorsion du profil thermique vertical
  """
  def __init__(self,sigma = 10.0,rho = 28.0, beta = 8.0/3.0):
    self.sigma = sigma
    self.rho = rho
    self.beta = beta
    self.name = "Lorenz"

  def deriv(self,state):
    """Calcule des dérivées dx/dt, dy/dt et dz/dt

    Args:
        state (ndarray):  derive des vecteurs ie state = [x,y,z] ou bien state = [[x1,y1,z1],
                                                                                  [x2,y2,z2]
                                                                                  [x3,y3,z3],...]
    Return: 
        dérivées de meme forme que state
    """
    x = state[...,0]  #
    y = state[...,1]
    z = state[...,2]

    dx = self.sigma*(y - x)
    dy = x*(self.rho - z) - y
    dz = x*y - self.beta*z

    return np.stack([dx, dy, dz], axis=-1)

  @property
  def default_state(self):
    return np.array([0.1, 0.0, 0.0])
  
  @property
  def fixed_points(self) -> list[np.ndarray]:
    """
    Points fixes analytiques du système de Lorenz.
    Utiles pour la section de Poincaré (plan z = rho - 1).
 
    Pour ρ > 1 :  C± = (±√(β(ρ−1)),  ±√(β(ρ−1)),  ρ−1)
    """

    if self.rho < 1:
      return [np.zeros(3)]
    val = np.sqrt(self.beta * (self.rho - 1))
    return [np.array([ val,  val, self.rho - 1]),
            np.array([-val, -val, self.rho - 1]),]

  @property
  def poincare_z(self):
    """Plan de coupe canonique pour la section de Poincaré."""
    return self.rho - 1.0


################
# Attracteur de Rössler
################
class RosslerSystem(ChaoticSysteme):
  pass


SYSTEMS: dict[str, ChaoticSystem] = {"Lorenz":  LorenzSystem(),"Rössler": RosslerSystem()}