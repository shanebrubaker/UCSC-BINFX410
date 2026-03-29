# BINFX410 · Chapter 10 · Connectomics Exercise

A complete end-to-end connectomics pipeline implemented as Jupyter notebooks,
taking students from raw simulated microscopy images all the way to a
biologically-constrained neural network.

---

## Prerequisites

```bash
pip install -r requirements.txt
```

All notebooks require Python ≥ 3.10. GPU is optional (PyTorch falls back to CPU).

---

## Notebook Sequence

Run notebooks **in order**. Each one saves data to `data/` for the next.

| # | Notebook | Topic | Key Output |
|---|----------|-------|------------|
| 1 | `01_image_simulation.ipynb` | Simulate fluorescence neuron images | `data/connectome_dataset.pkl` |
| 2 | `02_segmentation_cnn.ipynb` | U-Net CNN segmentation (PyTorch) | `data/segmentation_results.pkl` |
| 3 | `03_connectome_graph.ipynb` | Build NetworkX connectome graph | `data/connectome_graphs.pkl` |
| 4 | `04_connectome_analysis.ipynb` | Graph topology & motif analysis | `data/connectome_architecture.pkl` |
| 5 | `05_connectome_neural_network.ipynb` | Connectome-constrained PyTorch NN | trained model |

---

## Pipeline Overview

```
Notebook 1               Notebook 2                Notebook 3
─────────────────     ──────────────────────     ─────────────────────
Synthetic neuron  →   U-Net CNN detects soma  →  Assign synapses to
fluorescence images   & synaptic terminals        neuron pairs →
+ ground-truth masks  (PyTorch, Dice loss)         NetworkX DiGraph

Notebook 4               Notebook 5
────────────────────     ─────────────────────────────────
Graph topology:     →   ConnectomeNet (PyTorch):
adjacency matrix,       hidden-layer weights masked
random walk,            to match biological adjacency
motifs, clustering      matrix; compare to FreeNet
```

---

## Learning Objectives (cumulative)

1. Explain the connectomics pipeline: imaging → segmentation → graph → model
2. Generate realistic synthetic neuron microscopy images
3. Train a U-Net segmentation model in PyTorch (Dice + BCE loss, augmentation)
4. Build and query a directed weighted NetworkX graph from detected connections
5. Characterize a connectome using degree distribution, centrality, motifs, random walk
6. Design a biologically-constrained neural network whose weights mirror real synapses
7. Evaluate the benefit of sparse, structured connectivity over dense random networks

---

## Repository Structure

```
exercise-connectomics/
├── requirements.txt
├── README.md
├── utils/
│   ├── __init__.py
│   └── neuron_sim.py       ← Image synthesis library
├── data/                   ← Auto-created by notebooks
│   ├── connectome_dataset.pkl
│   ├── segmentation_results.pkl
│   ├── connectome_graphs.pkl
│   ├── connectome_architecture.pkl
│   └── unet_best.pt        ← Saved model weights
└── notebooks/
    ├── 01_image_simulation.ipynb
    ├── 02_segmentation_cnn.ipynb
    ├── 03_connectome_graph.ipynb
    ├── 04_connectome_analysis.ipynb
    └── 05_connectome_neural_network.ipynb
```

---

## Key Libraries

| Library | Role |
|---------|------|
| `torch` / `torchvision` | U-Net segmentation, connectome-constrained NN |
| `networkx` | Graph construction, topology analysis, visualization |
| `scikit-image` / `scipy` | Image processing, connected-component labeling |
| `numpy` / `matplotlib` | Numerics, visualization |
| `seaborn` | Statistical plotting |
| `pandas` | Tabular data |

---

## Student Exercise Summary

Each notebook contains **5 exercises** (4 standard + 1 challenge):

- **NB1**: Image noise, neuron density, axon morphology, synapse model limitations
- **NB2**: Skip connections, loss functions, model capacity, PR curve, axon segmentation
- **NB3**: Synapse assignment accuracy, distance threshold sweep, community detection, small-world test
- **NB4**: Adjacency symmetry, eigenvalue spectrum, leaky integrate-and-fire simulation, Louvain communities
- **NB5**: True sparse layers, biological task design, Hebbian plasticity, connectome topology vs. random, C. elegans

---

## Notes for Instructors

- **Notebook 2 training** takes ~2–5 minutes on CPU (30 epochs on 256×256 patches).
- **Notebook 5 ablation** trains 4 networks at 40 epochs each; ~5–10 min on CPU.
- The synthetic dataset is fully deterministic (seeded); students should get consistent results.
- Notebooks 3–5 gracefully handle cases where the CNN detects fewer neurons than expected.
