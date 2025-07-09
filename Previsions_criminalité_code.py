# ==============================================================
#  IMPORTATIONS
# ==============================================================

import numpy as np
from scipy.ndimage import gaussian_filter  # Pour flou gaussien
from scipy.signal import convolve2d        # Pour Laplacien 2D
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ==============================================================
#  PARAMÈTRES GLOBAUX DU MODÈLE
# ==============================================================

NB_TEMPS         = 100     # Nombre de pas de temps
PAS_DE_TEMPS     = 1.0     # Durée d’un pas de temps
TAILLE_GRILLE    = 50      # Grille carrée : TAILLE_GRILLE x TAILLE_GRILLE
NB_CRIMINELS     = 30      # Nombre de criminels au temps initial

# Paramètres du modèle d’attractivité dynamique (Short et al.)
A0_INIT          = 0.4     # Attractivité initiale
ETA              = 0.2     # Coefficient de diffusion spatiale
OMEGA            = 0.1     # Coefficient de décroissance naturelle
KAPPA            = 1.0     # Poids de la densité ρ

# ==============================================================
#  FONCTION 1 — SIMULATION DE LA DENSITÉ DE CRIMINELS ρ(t)
# ==============================================================

def simuler_densite_criminels(rho_initiale, sigma=1.0):
    """
    Simule l’évolution de la densité de criminels ρ(t) sur une grille carrée.

    1) Les criminels se déplacent de manière aléatoire (4-voisins)
    2) Probabilité de cambriolage p = 1 − exp(−A⋅dt)
    3) Les criminels disparus sont réinjectés selon A

    Paramètres :
    ------------
    rho_initiale : ndarray (TAILLE_GRILLE, TAILLE_GRILLE)
        État initial (entiers)

    sigma : float
        Écart-type du flou gaussien pour calculer l’attractivité locale.

    Retour :
    --------
    ndarray (NB_TEMPS+1, TAILLE_GRILLE, TAILLE_GRILLE)
        Historique temporel de ρ(t)
    """
    historique = []
    rho = rho_initiale.copy().astype(int)
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    for _ in range(NB_TEMPS):
        # 1) Calcul de l’attractivité instantanée A par flou gaussien
        attractivite = gaussian_filter(rho.astype(float), sigma=sigma)

        rho_suiv = np.zeros_like(rho)
        nb_disparus = 0

        # 2) Pour chaque cellule, mouvement ou cambriolage
        for i in range(TAILLE_GRILLE):
            for j in range(TAILLE_GRILLE):
                c = rho[i, j]
                if c == 0:
                    continue
                p_cambriolage = 1 - np.exp(-attractivite[i, j] * PAS_DE_TEMPS)

                for _ in range(c):
                    if np.random.rand() < p_cambriolage:
                        nb_disparus += 1
                    else:
                        di, dj = directions[np.random.randint(4)]
                        ni, nj = i + di, j + dj
                        if 0 <= ni < TAILLE_GRILLE and 0 <= nj < TAILLE_GRILLE:
                            rho_suiv[ni, nj] += 1
                        else:
                            rho_suiv[i, j] += 1  # rebond sur bord

        # 3) Réinjection des criminels disparus
        if nb_disparus > 0:
            poids = attractivite.ravel()
            poids = poids if poids.sum() > 0 else np.ones_like(poids)
            proba = poids / poids.sum()
            indices = np.random.choice(TAILLE_GRILLE * TAILLE_GRILLE,
                                       size=nb_disparus, p=proba)
            for k in indices:
                i, j = divmod(k, TAILLE_GRILLE)
                rho_suiv[i, j] += 1

        historique.append(rho.copy())
        rho = rho_suiv

    historique.append(rho.copy())
    return np.array(historique)

# ==============================================================
#  FONCTION 2 — CALCUL DE L’ATTRACTIVITÉ DYNAMIQUE B(t)
# ==============================================================

def calculer_attractivite_dynamique(rho_evolution):
    """
    Évolue B(t) selon :
        ∂B/∂t = KAPPA⋅ρ − OMEGA⋅B + ETA⋅∇²B

    Paramètre :
    -----------
    rho_evolution : ndarray (NB_TEMPS+1, TAILLE_GRILLE, TAILLE_GRILLE)
        Historique de ρ(t)

    Retour :
    --------
    ndarray (NB_TEMPS+1, TAILLE_GRILLE, TAILLE_GRILLE)
        Historique de B(t)
    """
    nb_etapes, n, _ = rho_evolution.shape
    B = np.zeros((n, n), dtype=float)
    historique_B = []

    laplacien = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])

    for t in range(nb_etapes - 1):
        rho = rho_evolution[t]

        B += KAPPA * rho * PAS_DE_TEMPS
        B += ETA * convolve2d(B, laplacien, mode='same', boundary='wrap') * PAS_DE_TEMPS
        B -= OMEGA * B * PAS_DE_TEMPS

        historique_B.append(B.copy())

    historique_B.append(B.copy())
    return np.array(historique_B)

# ==============================================================
#  FONCTION 3 — AFFICHAGE INTERACTIF
# ==============================================================

def afficher_rho_et_B(historique_rho, historique_B):
    """
    Affiche ρ(t) et B(t) côte à côte avec un slider pour naviguer dans le temps.
    """
    nb_etapes = historique_rho.shape[0]

    fig, (ax_rho, ax_B) = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(bottom=0.25)

    im_rho = ax_rho.imshow(historique_rho[0],
                           cmap='hot', vmin=0, vmax=historique_rho.max())
    im_B = ax_B.imshow(historique_B[0],
                       cmap='viridis', vmin=0, vmax=historique_B.max())

    for ax in (ax_rho, ax_B):
        ax.axis('off')
    ax_rho.set_title("Densité ρ(t)")
    ax_B.set_title("Attractivité B(t)")

    fig.colorbar(im_rho, ax=ax_rho, fraction=0.046, pad=0.04).set_label("Nombre de criminels")
    fig.colorbar(im_B, ax=ax_B, fraction=0.046, pad=0.04).set_label("Niveau d'attractivité")

    axe_slider = plt.axes([0.2, 0.08, 0.6, 0.03])
    slider = Slider(axe_slider, 'Temps', 0, nb_etapes - 1, valinit=0, valstep=1)

    def update(val):
        t = int(slider.val)
        im_rho.set_data(historique_rho[t])
        im_B.set_data(historique_B[t])
        fig.suptitle(f"t = {t}", fontsize=14)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.show()

# ==============================================================
#  SCRIPT PRINCIPAL
# ==============================================================

def main():
    """
    Lance la simulation complète et l’affichage interactif.
    """
    # Initialisation : criminels placés aléatoirement
    rho_initiale = np.zeros((TAILLE_GRILLE, TAILLE_GRILLE), dtype=int)
    indices_initiaux = np.random.choice(TAILLE_GRILLE * TAILLE_GRILLE,
                                        NB_CRIMINELS)
    for k in indices_initiaux:
        i, j = divmod(k, TAILLE_GRILLE)
        rho_initiale[i, j] += 1

    # Simulation
    historique_rho = simuler_densite_criminels(rho_initiale)
    historique_B = calculer_attractivite_dynamique(historique_rho)

    # Affichage
    afficher_rho_et_B(historique_rho, historique_B)

if __name__ == "__main__":
    main()
