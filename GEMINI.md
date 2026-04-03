# Guide du Projet LP-Diff pour l'Assistant IA (Gemini)

## 🎯 Objectif du Projet
Le projet **LP-Diff** (*Towards Improved Restoration of Real-World Degraded License Plate*) est un modèle d'Intelligence Artificielle de **super-résolution conditionnelle par diffusion**. Son but est de restaurer et d'améliorer la qualité d'images de plaques d'immatriculation dégradées en conditions réelles. Il a la particularité de s'appuyer sur une approche **multi-frames** (en fusionnant les informations de 3 images basse résolution) pour générer une image haute résolution claire et précise.

## 🏗️ Architecture et Logique Fonctionnelle

### Pipeline de Restauration
Le modèle ne restaure pas directement de LR (Basse Résolution) vers HR (Haute Résolution). Il utilise une logique en deux temps :
1. **MTA (Multi-Frame Temporal Attention)** : Fusionne les 3 frames LR (`LR1`, `LR2`, `LR3`) pour produire une prédiction brute (la "Condition").
2. **Diffusion (DDPM avec UNet)** : Apprend à restaurer le résidu manquant `(HR - Condition)`. Le processus de diffusion est conditionné par la sortie du MTA.

```text
3 LR frames (112×224) → MTA → Condition → Diffusion → SR (112×224)
                              ↓
                          HR (112×224) ← Double Loss
```

### Double Loss et Précision Mixte
Lors de l'entraînement (`optimize_parameters` dans `model.py`), le modèle minimise deux fonctions de perte simultanément, combinées avec un poids (`lambda_mta` souvent fixé à `1.0` dans la config) :
- **`loss_diffusion`** : Qualité finale de la restauration de l'image (débruitage).
- **`loss_mta`** : (Loss L1) Force le module MTA à produire une condition prédictive significative pour éviter l'effondrement du réseau.

> **Note d'optimisation** : Le calcul de la *loss* et le *forward pass* se font en **Mixed Precision** (`torch.amp.autocast` avec `bfloat16`) pour la rapidité et l'économie de VRAM. De plus, un *Gradient Clipping* (`max_norm=1.0`) est appliqué pour la stabilité avec un `CosineAnnealingLR`.

## 📂 Index des Fichiers Clés et Rôles

| Fichier / Dossier | Description et Rôle | Quand le consulter ou le modifier ? |
| :--- | :--- | :--- |
| `config/LP-Diff.json` | Fichier de configuration JSON. | Hyperparamètres (batch size, learning rate `1e-4`, `lambda_mta`), chemins (`dataroot`), ou architecture du UNet (ex: `in_channel: 6`, `attn_res: [16]`). |
| `run.py` | Script principal d'entraînement/validation. | Modifier la boucle d'entraînement, le logging (Tensorboard/WandB), et la logique de sauvegarde (`save_checkpoint_freq`). |
| `infer.py` | Script d'inférence. | Pour tester un modèle et générer des images pour l'évaluation. |
| `eval.py` | Outil de calcul des métriques. | Calcul du PSNR et SSIM sur le jeu de test final. |
| `data/LRHR.py` | Dataloader du dataset (`LRHRDataset`). | Structure MDLP (Dossiers `inputs/` et `gt/` par plaque). Les Tenseurs de sortie (`LR1`, `LR2`, `LR3`, `HR`) ont la *shape* `(B, 3, 112, 224)`. |
| `model/model.py` | Classe d'interface `DDPM` (`BaseModel`). | Très important. Gère `feed_data()`, `optimize_parameters()` (Double Loss, Mixed Precision) et `test()` (inférence). |
| `model/networks.py` | Définition et instanciation du Graphe. | Rattache le UNet et la logique `GaussianDiffusion` via `define_G`. |
| `model/LPDiff_modules/`| Dossier du cœur du modèle IA. | - `unet.py` : Réseau de débruitage conditionnel (entrée en 6 canaux: 3 bruit + 3 condition).<br>- `Multi_tmp_fusion.py` : MTA (Encodeur, Cross-Attention, Décodeur).<br>- `diffusion.py` : Processus de diffusion. |
| `viewer/` | Outil web local d'inspection. | Observer les résultats générés de manière interactive (slider HR / LR / SR). |

## 💻 Commandes Utiles

- **Entraînement sur un seul GPU** :
  ```bash
  python run.py -p train -c ./config/LP-Diff.json -gpu 0
  ```
- **Entraînement multi-GPU (DDP)** :
  ```bash
  torchrun --nproc_per_node=2 run.py -p train -c ./config/LP-Diff.json -gpu 0,1
  ```
- **Validation** :
  ```bash
  python run.py -p val -c ./config/LP-Diff.json -gpu 0
  ```
- **Inférence pure** :
  ```bash
  python infer.py -c ./config/LP-Diff.json -gpu 0
  ```

## 💡 Astuces Techniques et Pièges à Éviter
1. **Gestion des Datasets** : Le Dataloader s'attend à une structure où chaque sous-dossier de `inputs` contient des images nommées `img_0.jpg`, `img_1.jpg`, etc. Vérifiez l'expression régulière `extract_number` dans `LRHR.py` en cas d'erreur.
2. **Dimension des Tenseurs (Debugging)** :
   - `LR1`, `LR2`, `LR3`, `HR` et `Condition` (sortie du MTA) sont de forme `(B, 3, 112, 224)`.
   - À l'entrée du UNet, la condition et l'image bruitée sont concaténées, donnant un tenseur de forme `(B, 6, 112, 224)`. Le paramètre `"in_channel": 6` dans le `LP-Diff.json` est donc **strictement nécessaire**.
3. **Entraînement Distribué (DDP)** : Le code supporte `torchrun`. En mode distribué, **seul le rank 0** (GPU principal) effectue le logging, les calculs de validation globaux et sauvegarde les checkpoints.
4. **WandB Logging** : Le suivi des expériences est fortement intégré avec Weights & Biases via `core/wandb_logger.py`. Les métriques sont mises à jour dans `self.log_dict` (`l_pix`, `l_diffusion`, `l_mta`).
5. **Bruit et Scheduling** : Le `set_new_noise_schedule` (dans `model.py` et `diffusion.py`) permet d'avoir des paramètres (ex: `n_timestep`, `linear_start`, `linear_end`) distincts pour le `train` et la `val`.
