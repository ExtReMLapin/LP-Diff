# LP-Diff Project Guidelines

## Architecture Overview

LP-Diff est un modèle de **super-résolution conditionnelle par diffusion** pour la restauration de plaques d'immatriculation dégradées. Il utilise **3 frames basse résolution** en entrée pour produire une image haute résolution.

### Pipeline de Dégradation-Restauration

```
3 LR frames (112×224) → MTA → Condition → Diffusion → SR (112×224)
                              ↓
                          HR (112×224) ← Loss
```

### Composants Clés

| Module | Fichier | Description |
|--------|---------|-------------|
| **MTA** | `model/LPDiff_modules/Multi_tmp_fusion.py` | Multi-Frame Temporal Attention: fusionne 3 frames LR via encodeur, cross-attention, GCA, décodeur |
| **UNet** | `model/LPDiff_modules/unet.py` | Réseau de débruitage conditionnel avec attention spatiale |
| **Diffusion** | `model/LPDiff_modules/diffusion.py` | Processus de diffusion conditionnelle (GaussianDiffusion) |
| **DDPM** | `model/model.py` | Wrapper modèle avec optimisation, logging, checkpointing |

## Principes de Conception

### 1. Conditionnement par MTA

Le modèle ne restaure pas directement de LR → HR. Il apprend:
- **MTA**: Fusion temporelle des 3 frames → prédiction brute
- **Diffusion**: Restauration du résidu `(HR - MTA_output)`

```python
# diffusion.py:244-248
condition = self.MTA(x_in['LR1'], x_in['LR2'], x_in['LR3'])
x_start = x_in['HR'] - condition  # Résidu à restaurer
```

### 2. Double Loss

```python
# diffusion.py:273-274
loss_diffusion = self.loss_func(noise, x_recon)
loss_mta = F.l1_loss(condition, x_in['HR'], reduction='sum')
return loss_diffusion + self.lambda_mta * loss_mta, ...
```

- **loss_diffusion**: Qualité de la restauration finale
- **loss_mta**: Force MTA à produire une condition significative (évite l'effondrement)

### 3. Mixed Precision Training

```python
# model/model.py:78-79
with autocast(device_type='cuda', dtype=torch.bfloat16):
    l_pix, l_diffusion, l_mta = self.netG(self.data)
```

## Structure des Données

### Dataset MDLP

```
dataset_plaques/
├── train/
│   ├── inputs/     # 3 frames LR par plaque
│   │   └── {plate_id}/img_0.jpg, img_1.jpg, ...
│   └── gt/         # Image HR par plaque
│       └── {plate_id}/img_0.jpg, ...
└── val/            # Même structure
```

### Format des Tensors

| Tensor | Shape | Description |
|--------|-------|-------------|
| LR1/2/3 | (B, 3, 112, 224) | 3 frames basse résolution |
| HR | (B, 3, 112, 224) | Ground truth haute résolution |
| Condition | (B, 3, 112, 224) | Sortie MTA |

## Hyperparamètres Critiques

### Configuration (`config/LP-Diff.json`)

```json
{
  "model": {
    "unet": {
      "in_channel": 6,    // 3 (condition MTA) + 3 (noisy)
      "out_channel": 3,
      "inner_channel": 64,
      "channel_multiplier": [1, 2, 4, 8, 8],
      "attn_res": [16],   // Attention à resolution 16
      "res_blocks": 2
    },
    "beta_schedule": {
      "train": {
        "schedule": "linear",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 1e-2
      }
    }
  },
  "train": {
    "n_iter": 1000000,
    "optimizer": {"type": "adam", "lr": 1e-4},
    "lambda_mta": 1.0  // Poids loss MTA
  }
}
```

## Points d'Attention pour les Modifications

### 1. Modifier l'Architecture MTA

Fichier: `model/LPDiff_modules/Multi_tmp_fusion.py`

```python
# Lignes 520-556: Classe MTA
# Composants modifiables:
# - Encoder: CNN 4 couches (3→16→32→64→64)
# - CrossAttentionLayer: 8 heads, dim_feedforward=256
# - GradientCurvatureAttention: attention par gradients
# - IntraframeAtt: fusion canal+spatial
# - Decoder: ConvTranspose2d (64→32→16→3)
```

### 2. Modifier l'UNet

Fichier: `model/LPDiff_modules/unet.py`

```python
# Lignes 162-262: Classe UNet
# Paramètres modifiables:
# - channel_mults: facteurs d'expansion des channels
# - attn_res: résolutions avec attention
# - res_blocks: blocs résiduels par niveau
```

### 3. Modifier le Processus de Diffusion

Fichier: `model/LPDiff_modules/diffusion.py`

```python
# Lignes 74-277: Classe GaussianDiffusion
# Points critiques:
# - p_losses(): calcul des losses (lignes 243-274)
# - p_sample_loop(): inférence (lignes 196-221)
# - set_new_noise_schedule(): schedule beta (lignes 110-157)
```

### 4. Modifier l'Entraînement

Fichier: `run.py`

```python
# Boucle d'entraînement: lignes 121-161
# Validation: lignes 162-258
# Points modifiables:
# - save_checkpoint_freq: fréquence sauvegarde
# - val_warmup_epochs: warmup avant validation
# - print_freq: fréquence logging
```

## Commandes de Référence

```bash
# Entraînement single GPU
python run.py -p train -c ./config/LP-Diff.json -gpu 0

# Entraînement multi-GPU
torchrun --nproc_per_node=2 run.py -p train -c ./config/LP-Diff.json -gpu 0,1

# Validation
python run.py -p val -c ./config/LP-Diff.json -gpu 0

# Inférence
python infer.py
```

## Debugging

### Vérifier les Shapes

```python
# Après feed_data
print(data['LR1'].shape)  # (B, 3, 112, 224)
print(data['HR'].shape)    # (B, 3, 112, 224)

# Après MTA
condition = model.MTA(lr1, lr2, lr3)
print(condition.shape)  # (B, 3, 112, 224)

# Entrée UNet
input_unet = torch.cat([condition, x_noisy], dim=1)
print(input_unet.shape)  # (B, 6, 112, 224)
```

### Logs d'Entraînement

```
<epoch:  45, iter:    900000> l_pix: 1.2340e-02 l_diffusion: 1.1230e-02 l_mta: 1.1100e-02 l_total: 3.4540e-02 lr: 5.0000e-05
# Validation # PSNR: 1.4400e+01  val_loss: 1.2340e-02
```

## Métriques

| Métrique | Formule | Objectif |
|----------|---------|----------|
| **PSNR** | 10·log₁₀(255²/MSE) | ↑ Plus haut = meilleur |
| **SSIM** | Structure similarity | ↑ Plus proche de 1 = meilleur |
| **NED** | Normalized Edit Distance | ↓ Plus bas = meilleur OCR |
| **ACC** | Text Recognition Accuracy | ↑ Plus haut = meilleur OCR |

## Notes Importantes

1. **Gradient Clipping**: `max_norm=1.0` pour stabilité
2. **LR Scheduler**: CosineAnnealingLR de 1e-4 à 1e-6
3. **Checkpointing**: Sauvegarde toutes les 10k itérations
4. **Validation**: Tous les epochs après warmup (5 epochs)
5. **Distributed**: Rank 0 seul fait le logging/checkpointing
