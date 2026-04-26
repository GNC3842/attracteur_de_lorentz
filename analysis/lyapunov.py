import numpy as np
from simulation.integrator import integrate

def calculer_lyapunov(systeme, dt, nb_pas, warmup=1000):
    """
    Calcule l'exposant de Lyapunov d'un système donné.
    
    Le principe est de suivre deux trajectoires : 
    - Une trajectoire 'A' (normale)
    - Une trajectoire 'B' (très légèrement décalée au début)
    On regarde à chaque étape à quel point elles s'écartent.
    """
    
    # 1. Préparation des conditions initiales
    # On commence au point par défaut du système
    etat_a = systeme.default_state
    
    # Phase de chauffe (warmup) : 
    # On fait tourner le système un peu pour être sûr d'être sur l'attracteur
    for _ in range(warmup):
        # Un pas d'intégration (on utilise RK4 ou une méthode simple)
        etat_a = integrate(systeme, etat_a, dt)
        
    # 2. Création de la trajectoire perturbée (B)
    # On crée un écart minuscule (d0)
    d0 = 1e-8
    etat_b = np.copy(etat_a)
    etat_b[0] += d0 # On décale juste un tout petit peu sur l'axe X
    
    somme_lyap = 0.0
    historique_lyap = []

    # 3. Boucle principale de calcul
    for i in range(1, nb_pas + 1):
        # On fait avancer les deux trajectoires
        etat_a = integrate(systeme, etat_a, dt)
        etat_b = integrate(systeme, etat_b, dt)
        
        # On mesure la nouvelle distance entre A et B
        diff = etat_b - etat_a
        d1 = np.sqrt(np.sum(diff**2)) # Norme du vecteur (distance)
        
        # On calcule le taux d'écartement : log(distance_finale / distance_initiale)
        # Si d1 > d0, le log est positif -> Chaos
        somme_lyap += np.log(d1 / d0)
        
        # L'exposant de Lyapunov est la moyenne de ces écarts
        lyap_actuel = somme_lyap / (i * dt)
        historique_lyap.append(lyap_actuel)
        
        # --- ETAPE CRUCIALE : LA RENORMALISATION ---
        # On ramène la trajectoire B tout près de A pour éviter qu'elle 
        # ne s'enfuie à l'autre bout de l'espace. 
        # On garde par contre la DIRECTION de l'écart.
        etat_b = etat_a + (diff / d1) * d0
        
    return historique_lyap


# --- EXEMPLE D'UTILISATION ---
# lorenz = LorenzSystem()
# resultats = calculer_lyapunov(lorenz, dt=0.01, nb_pas=10000)
# print(f"L'exposant final est : {resultats[-1]}")