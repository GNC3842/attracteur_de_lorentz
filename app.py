"""
app.py — Point d'entrée pour tester le projet Lorenz
=====================================================
Lance ce script depuis la racine du projet :

    python app.py

Il teste chaque module dans l'ordre et affiche les résultats.
Les graphes s'ouvrent dans des fenêtres matplotlib séparées.
Ferme chaque fenêtre pour passer au test suivant.
"""

import sys
import traceback
import numpy as np
import matplotlib.pyplot as plt

# ─── Couleurs terminal ────────────────────────────────────────
OK   = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
SKIP = "\033[93m~\033[0m"
HEAD = "\033[1;94m"
RST  = "\033[0m"

def section(title):
    print(f"\n{HEAD}{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}{RST}")

def success(msg):  print(f"  {OK}  {msg}")
def failure(msg):  print(f"  {FAIL}  {msg}")
def skipped(msg):  print(f"  {SKIP}  {msg}")


# ════════════════════════════════════════════════════════════════
#  1. IMPORTS — vérifie que tous les modules se chargent
# ════════════════════════════════════════════════════════════════
section("1 · Imports")

errors = {}

try:
    from simulation.system import LorenzSystem, RosslerSystem, SYSTEMS
    success("simulation.system     (LorenzSystem, RosslerSystem, SYSTEMS)")
except Exception as e:
    failure(f"simulation.system     → {e}")
    errors["system"] = e

try:
    from simulation.integrator import integrate, multi_integrate, compute_speed
    success("simulation.integrator (integrate, integrate_ensemble, compute_speed)")
except Exception as e:
    failure(f"simulation.integrator → {e}")
    errors["integrator"] = e

try:
    from visualization.phase_space import plot_trajectory_3d, plot_ensemble_3d
    success("visualization.phase_space (plot_trajectory_3d, plot_ensemble_3d)")
except Exception as e:
    failure(f"visualization.phase_space → {e}")
    errors["phase_space"] = e

try:
    from analysis.lyapunov import lyapunov_exponent
    success("analysis.lyapunov     (lyapunov_exponent)")
except Exception as e:
    skipped(f"analysis.lyapunov     → pas encore implémenté ({e})")
    errors["lyapunov"] = e

try:
    from analysis.poincare import poincare_section, poincare_return_map
    success("analysis.poincare     (poincare_section, poincare_return_map)")
except Exception as e:
    skipped(f"analysis.poincare     → pas encore implémenté ({e})")
    errors["poincare"] = e

try:
    from visualization.diagnostics import (
        plot_time_series, plot_divergence, plot_lyapunov, plot_poincare
    )
    success("visualization.diagnostics (plot_time_series, ...)")
except Exception as e:
    skipped(f"visualization.diagnostics → pas encore implémenté ({e})")
    errors["diagnostics"] = e


# ════════════════════════════════════════════════════════════════
#  2. SIMULATION — intégration de base
# ════════════════════════════════════════════════════════════════
section("2 · Simulation")

if "system" in errors or "integrator" in errors:
    failure("Modules manquants — section ignorée")
else:
    DT      = 0.005
    N_STEPS = 10_000

    # Système par défaut
    system = LorenzSystem()
    print(f"  Système  : {system.name}  σ={system.sigma}  ρ={system.rho}  β={system.beta:.4f}")

    # Intégration
    try:
        traj = integrate(system, system.default_state, N_STEPS, DT)
        assert traj.shape == (N_STEPS, 3), f"Forme inattendue : {traj.shape}"
        success(f"integrate()           → {traj.shape}  (durée = {N_STEPS * DT:.1f} u.t.)")
    except Exception as e:
        failure(f"integrate()           → {e}")
        traj = None

    # Vitesse
    if traj is not None:
        try:
            speed = compute_speed(traj)
            assert speed.min() >= 0 and speed.max() <= 1
            success(f"compute_speed()       → [{speed.min():.3f}, {speed.max():.3f}]  (normalisé)")
        except Exception as e:
            failure(f"compute_speed()       → {e}")

    # Points fixes
    try:
        fps = system.fixed_points
        success(f"fixed_points          → {len(fps)} points  |  C+ ≈ ({fps[0][0]:.2f}, {fps[0][1]:.2f}, {fps[0][2]:.2f})")
    except Exception as e:
        skipped(f"fixed_points          → attribut absent ({e})")

    # Ensemble vectorisé
    try:
        N_ENS   = 50
        rng     = np.random.default_rng(0)
        states0 = system.default_state + rng.normal(0, 1e-3, (N_ENS, 3))
        ens     = multi_integrate(system, states0, 2000, DT)
        assert ens.shape == (2000, N_ENS, 3)
        success(f"integrate_ensemble()  → {ens.shape}  ({N_ENS} trajectoires)")
    except Exception as e:
        failure(f"integrate_ensemble()  → {e}")
        ens = None

    # Rössler
    try:
        ros   = RosslerSystem()
        traj_r = integrate(ros, ros.default_state, 5000, DT)
        success(f"RosslerSystem         → {traj_r.shape}")
    except Exception as e:
        skipped(f"RosslerSystem         → {e}")


# ════════════════════════════════════════════════════════════════
#  3. ANALYSE — Lyapunov & Poincaré
# ════════════════════════════════════════════════════════════════
section("3 · Analyse")

lyap_exps = None

if "lyapunov" in errors:
    skipped("lyapunov.py non disponible — à implémenter")
else:
    try:
        lyap_exps = lyapunov_exponent(system, system.default_state,
                                       n_steps=15_000, dt=DT)
        lval = lyap_exps[-1]
        verdict = "CHAOS ✓" if lval > 0 else "STABLE"
        success(f"lyapunov_exponent()   → λ = {lval:.4f}  ({verdict})  "
                f"[attendu ≈ 0.9]")
    except Exception as e:
        failure(f"lyapunov_exponent()   → {e}")
        traceback.print_exc()

poincare_pts = None

if "poincare" in errors:
    skipped("poincare.py non disponible — à implémenter")
else:
    try:
        z_plane      = system.poincare_z if hasattr(system, "poincare_z") else 27.0
        poincare_pts = poincare_section(traj, z_plane=z_plane)
        success(f"poincare_section()    → {len(poincare_pts)} points  "
                f"(plan z = {z_plane:.1f})")
    except Exception as e:
        failure(f"poincare_section()    → {e}")

    if poincare_pts is not None and len(poincare_pts) >= 2:
        try:
            xn, xn1 = poincare_return_map(traj, z_plane=z_plane)
            success(f"poincare_return_map() → {len(xn)} paires (xn, xn+1)")
        except Exception as e:
            failure(f"poincare_return_map() → {e}")


# ════════════════════════════════════════════════════════════════
#  4. VISUALISATION
# ════════════════════════════════════════════════════════════════
section("4 · Visualisation")
if traj is None:
    failure("Trajectoire non disponible — visualisation ignorée")
else:
    # ── 4a. Phase 3D ──────────────────────────────────────────
    if "phase_space" not in errors:
        try:
            print("  Ouverture fenêtre 1/3 : trajectoire 3D…")
            fig1 = plot_trajectory_3d(
                traj,
                title=f"Attracteur de Lorenz  σ={system.sigma}  ρ={system.rho}  β={system.beta:.2f}",
            )
            plt.show()
            success("plot_trajectory_3d()  OK")
        except Exception as e:
            failure(f"plot_trajectory_3d()  → {e}")
            traceback.print_exc()

        if ens is not None:
            try:
                print("  Ouverture fenêtre 2/3 : ensemble…")
                fig2 = plot_ensemble_3d(ens, title="Dispersion d'ensemble (N=50, ε=1e-3)")
                plt.show()
                success("plot_ensemble_3d()    OK")
            except Exception as e:
                failure(f"plot_ensemble_3d()    → {e}")
    else:
        skipped("phase_space.py non disponible")

    # ── 4b. Diagnostics ───────────────────────────────────────
    if "diagnostics" not in errors:
        try:
            print("  Ouverture fenêtre 3/3 : diagnostics…")
            fig_grid, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig_grid.suptitle("Diagnostics — Attracteur de Lorenz",
                              fontsize=13, color="#0d2340")

            # Séries temporelles (x uniquement pour la lisibilité)
            t_arr = np.arange(len(traj)) * DT
            axes[0, 0].plot(t_arr, traj[:, 0], color="#00e5ff", lw=0.7)
            axes[0, 0].set_title("x(t)", fontsize=10)
            axes[0, 0].set_xlabel("t"); axes[0, 0].grid(alpha=0.3)

            axes[0, 1].plot(t_arr, traj[:, 2], color="#69ff47", lw=0.7)
            axes[0, 1].set_title("z(t)", fontsize=10)
            axes[0, 1].set_xlabel("t"); axes[0, 1].grid(alpha=0.3)

            # Lyapunov si disponible
            if lyap_exps is not None:
                t_lyap = np.arange(1, len(lyap_exps) + 1) * DT
                axes[1, 0].plot(t_lyap, lyap_exps, color="#ff6e40", lw=0.8)
                axes[1, 0].axhline(lyap_exps[-1], color="white", lw=1,
                                   linestyle="--", alpha=0.6,
                                   label=f"λ = {lyap_exps[-1]:.3f}")
                axes[1, 0].set_title("Exposant de Lyapunov", fontsize=10)
                axes[1, 0].set_xlabel("t"); axes[1, 0].legend(fontsize=8)
                axes[1, 0].grid(alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, "lyapunov.py\nnon disponible",
                                ha="center", va="center",
                                transform=axes[1, 0].transAxes,
                                fontsize=11, color="gray")
                axes[1, 0].set_title("Exposant de Lyapunov", fontsize=10)

            # Poincaré si disponible
            if poincare_pts is not None and len(poincare_pts) > 0:
                axes[1, 1].scatter(poincare_pts[:, 0], poincare_pts[:, 1],
                                   s=5, alpha=0.7,
                                   c=np.arange(len(poincare_pts)),
                                   cmap="plasma")
                axes[1, 1].set_title(f"Section de Poincaré (z={z_plane:.1f})",
                                     fontsize=10)
                axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("y")
                axes[1, 1].grid(alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, "poincare.py\nnon disponible",
                                ha="center", va="center",
                                transform=axes[1, 1].transAxes,
                                fontsize=11, color="gray")
                axes[1, 1].set_title("Section de Poincaré", fontsize=10)

            fig_grid.patch.set_facecolor("#07080f")
            for ax in axes.flat:
                ax.set_facecolor("#0d1220")
                ax.tick_params(colors="#8899cc")
                for spine in ax.spines.values():
                    spine.set_color("#1a1f35")

            plt.tight_layout()
            plt.show()
            success("Graphe diagnostics    OK")
        except Exception as e:
            failure(f"Graphe diagnostics    → {e}")
            traceback.print_exc()
    else:
        skipped("diagnostics.py non disponible")


# ════════════════════════════════════════════════════════════════
#  5. BILAN
# ════════════════════════════════════════════════════════════════
section("5 · Bilan")

modules_done    = {"system", "integrator", "phase_space"} - errors.keys()
modules_missing = {"lyapunov", "poincare", "diagnostics"} & errors.keys()

for m in sorted(modules_done):
    success(f"{m:25s} opérationnel")

for m in sorted(modules_missing):
    skipped(f"{m:25s} à implémenter")

for m in sorted({"system", "integrator", "phase_space"} & errors.keys()):
    failure(f"{m:25s} erreur à corriger")

print()
if not modules_missing:
    print(f"  {OK}  Projet complet — prêt pour le notebook.")
else:
    print(f"  Prochaine étape : implémenter {', '.join(sorted(modules_missing))}")
print()