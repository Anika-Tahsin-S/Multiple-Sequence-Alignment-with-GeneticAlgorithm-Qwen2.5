# Sequence-Alignment-with-GeneticAlgorithm-Qwen2.5

Lightweight GA-based multiple-sequence alignment (MSA) with GPU-accelerated fitness (PyTorch) on Kaggle’s “Sequence Alignment (Bioinformatics) Dataset.” Uses affine gaps with PAM250, tournament selection, residue-count–preserving crossover, and improving mutations. Computes entropy/gap/identity metrics, draws heatmaps & dendrograms, writes CLUSTAL/FASTA. A local LLM (Qwen2.5-1.5B-Instruct via llama-cpp) produces a JSON report (conserved blocks, gap clusters, closest/divergent pairs). No API keys needed.

# Pipeline:

- Loads protein FASTA (+ PAM250 from Biopython; falls back to provided PAM250.txt, else BLOSUM62).
- Pads sequences to max length; initializes a population of gapped candidates.
- GA training (GPU fitness): sum-of-pairs with affine gap penalties (open/extend) using PyTorch, tournament selection, residue-count–preserving 1–2 point crossover, and local improving mutations (single slide / small block shift). Elitism + immigrants per generation.
- Saves best alignment to CLUSTAL (ga_best.aln) and FASTA (ga_best.fasta).
- Evaluates: per-column Shannon entropy (conservation), gap density, pairwise % identity matrix; optional UPGMA dendrogram; simple consensus.
- LLM analysis (no API): prompts Qwen2.5-1.5B-Instruct (GGUF) via llama-cpp to return strict JSON. A validator fixes 0/1-based ranges, verifies conserved/gap blocks by thresholds, and recomputes identity pairs.


*Kaggle:* https://www.kaggle.com/code/anikatahsin8/sequence-alignment-with-genetic-algorithm-qwen2-5


# Acknowledgements

Dataset: Sequence Alignment (Bioinformatics) Dataset (Kaggle); https://www.kaggle.com/datasets/samira1992/sequence-alignment-bioinformatics-dataset/data

Model: Qwen/Qwen2.5-1.5B-Instruct (GGUF); https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

Libraries: Biopython, PyTorch, matplotlib, llama-cpp-python

# Trained on
Kaggle Accelerator = GPU (Tesla T4)
(Works on CPU; GPU strongly recommended for the GA fitness step.)

# Prerequisites

**Core scientific**

- numpy>=1.26
- pandas>=2.0
- matplotlib>=3.7
- scipy>=1.10 (optional; for dendrogram)

**Bio**
- biopython>=1.85

**GPU / fitness**
- torch>=2.1 (CUDA-enabled build for GPU acceleration)

**Local LLM (no API)**

One of:
- llama-cpp-python (CPU only)
- llama-cpp-python-cu121 / llama-cpp-python-cu122 (GPU builds; set n_gpu_layers>0)

- Model: qwen2.5-1.5b-instruct-q4_k_m.gguf; (Download helper) huggingface_hub>=0.23 or huggingface-cli binary


# Outputs & interpretation

- Alignment: ga_best.aln (CLUSTAL), ga_best.fasta

**Figures:**

- entropy.png — per-column conservation (lower = more conserved)
- gap_density.png — per-column gap fraction
- pid_heatmap.png — pairwise % identity heatmap
- upgma_dendrogram.png — clustering (optional; needs SciPy)

**Tables:**
- pairwise_identities.csv

**Reports:**
- consensus.txt
- ga_llm_corrected.json — LLM JSON summary, validated vs. alignment
(fixes 0- vs 1-based ranges, checks entropy/gap thresholds, re-computes identity pairs)