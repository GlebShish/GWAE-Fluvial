# GWAE-Fluvial

training.ipynb – Demonstrates training of a Graph-based Wasserstein Autoencoder (GWAE) to learn latent representations of geological models, enabling realistic reconstruction. https://cloud.hw.tpu.ru/index.php/s/9iyKN6nY4j3FEpc - link to the training dataset

es.ipynb – Implements Assisted History Matching using a pre-trained GWAE and Evolution Strategies to find geological models in latent space for production data calibration under geological uncertainty.

## training.ipynb 
The notebook demonstrates the training process of a Graph-based Wasserstein Autoencoder (GWAE) for generating geological models under uncertainty. The GWAE is a graph-based variational autoencoder architecture designed to handle different geological scenarios (e.g., models with varying numbers of channels) by learning a low-dimensional latent space representation
researchgate.net. Unlike traditional grid-based approaches, this model uses graph convolutional neural networks to encode and decode reservoir models, allowing it to preserve geological realism and spatial relationships in the data. The latent space of the GWAE can be used to implicitly control geological realism (for example, through geodesic interpolation between latent points) and to analyze the variability of geological scenarios via techniques like PCA or t-SNE. 

Workflow Summary:
Data Loading: The dataset (a list of graph objects) is loaded from a pickle file. Each graph represents a 3D channelized reservoir model with node features such as porosity and permeability.
Preprocessing: Porosity and permeability features are normalized. Porosity is scaled to [-1, 1] and permeability (originally log-normally distributed) is transformed to a normal distribution for stable training.
Model Setup: Key hyperparameters are defined (latent dimension, learning rates, etc.). A PyTorch Geometric DataLoader is prepared for batching the graphs. The GWAE model and trainer are initialized. The model employs a Wasserstein AutoEncoder approach using a Maximum Mean Discrepancy (MMD) loss (with an RBF kernel) to match the latent distribution to a prior.
Training: The trainer can run two training phases: one for the autoencoder's main parameters (µ training for a number of epochs) and an optional fine-tuning for the MMD kernel width (σ training). Training metrics are logged using TensorBoard's SummaryWriter. (In this notebook, a pre-trained model state is loaded for demonstration. To reproduce training from scratch, you can skip loading the weights and train for the specified epochs.)
Evaluation: The trained model is tested on the dataset (e.g., reconstructing the input graphs). Predictions and true values are collected for analysis.
Visualization: The reconstruction performance is visualized. For instance, the code uses custom visual_tools to plot how well the GWAE reconstructs properties like porosity and permeability across the reservoir model. This helps verify that the GWAE preserves petrophysical relationships and spatial structures in the generated models

Below is a list of all major dependencies required to run this notebook. These can be saved into a requirements.txt file for easy installation. Ensure that the versions of PyTorch and PyTorch Geometric are compatible (the combination should be chosen based on the system and CUDA availability):
graphql
numpy
matplotlib
torch
torch-geometric
tqdm
scikit-learn
tensorboard       # for using TensorBoard with PyTorch (SummaryWriter)
dash              # for visualization tools (if using interactive Dash plots)
dash-html-components
dash-core-components
Note: The dash-html-components and dash-core-components packages are listed because the custom visual_tools.py uses them (as indicated by the warnings). In Dash 2.x, these are included in the main dash package, but installing them separately ensures backward compatibility with the code. If you intend to run the interactive visualization tools, having dash installed is necessary. If not using Dash-based visuals, those can be omitted. The core training and evaluation will work with just NumPy, Matplotlib, PyTorch (+torch-geometric), tqdm, and scikit-learn.


## es.ipynb 
The notebook Overview (Purpose, Methodology, and Outputs)
Purpose: This notebook demonstrates an Assisted History Matching (AHM) under geological uncertainty using a Graph-based Variational Autoencoder (GVAE) and Evolution Strategies (ES) optimization. It is a companion to the paper "History Matching under Uncertainty of Geological Scenarios with Implicit Geological Realism Control with Generative Deep Learning and Graph Convolutions" by Shishaev et al., 2025 (https://arxiv.org/abs/2507.10201). The goal is to calibrate a geological model (specifically a channelized reservoir) to match production data while maintaining geological realism. In simpler terms, the notebook shows how to take an observed production history (from a true but unknown geology) and find a plausible geological scenario that reproduces that history. The uniqueness here is the use of a latent space of a deep generative model to represent geological scenarios, and guiding the search in that space rather than directly in high-dimensional model parameters. This approach constrains the search to geologically reasonable scenarios (the GVAE ensures any latent vector decoded is a realistic geology) and uses a learned geodesic latent metric to penalize unrealistic departures. 

Methodology:
Data & Model Setup: A synthetic dataset of 3D channelized reservoir models is loaded. Each model is represented as a graph of ~1919 active cells (nodes) with two features (porosity and permeability). A Graph VAE (denoted as GVWAE in code) has been pre-trained on these models (R-GWAE_10_1919_30_mmd_1_2_1000_mu-500_si-500_Channels_1-3.pt). This VAE compresses each geological model into a 30-dimensional latent vector (z) and can decode any z back to a plausible geology. The VAE also provides a metric in latent space (via trainer.model.metric and get_measure) that quantifies how far a latent point is from the training manifold, enabling implicit realism control.

True Scenario and Simulation: One particular realization (index 1945) is designated as the "true" scenario. Its properties are fed into a reservoir simulator (tNavigator) to generate "observed" well production data (oil rates, water rates, etc.). The notebook parses this output (BASE_RES) and treats it as the target history to match.
Objective Function: The history-matching objective is multi-faceted:
Static data mismatch: Differences in rock properties at well locations (e.g., porosity at wells) between the candidate model and the true model.
Dynamic data mismatch: Differences in production/injection time-series between the candidate model’s simulation and the observed data.
Geological realism penalty: A term derived from the VAE’s latent-space metric (based on geodesic distance on the manifold of learned geological patterns) that penalizes unrealistic models.
These are combined into a single fitness score (the code uses a weighted sum, effectively giving equal weight to the raw summed squared errors for simplicity). The evolutionary algorithm maximizes this fitness (i.e., minimizes all mismatches and the penalty).

Evolutionary Optimization: A CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is employed to search the 30-D latent space for an optimal z. The initial population is centered around the mean of the training latent distribution (zs_mean), ensuring we start in a region of latent space that produces geologically plausible fields. Each generation, 51 candidates are decoded by the VAE into reservoir models, simulated, and evaluated against observed data. CMA-ES then updates the latent search distribution based on these evaluations. Over successive generations, the population’s best solution approaches one that yields a good history match. Logging to Weights & Biases tracks the progress of static and dynamic error reduction.

Result Extraction: After the iterations, the best-found latent vector is decoded to a reservoir model. The code runs one final simulation for this model and compares its production profiles to the true history. The resulting match is stored, and the model’s properties are saved (both in graph form and converted back to a 3D grid file). The notebook also captures an ensemble of top solutions and their production profiles, indicating the degree of non-uniqueness or uncertainty in the match (multiple models might fit the data nearly equally well).

Outputs:
Optimized Model: The primary result is the latent vector (1945_latent_code.pickle) and corresponding geological model that best matches the production data. This model’s porosity and permeability fields are saved to files (Generated_grids/_Geodesic_poro/perm), and its simulated well production closely overlaps the true scenario’s production.
History Match Quality: A DataFrame (1945_productions_opt_and_ref.pickle) is saved, containing the time-series of key production metrics for both the optimized model and the reference. One can inspect these to see the quality of the match. Ideally, the curves for oil rate, water rate, etc., from the optimized model align with those from the true model, indicating a successful history match.
Performance Metrics: The W&B logs (and prints) show the decline of misfit over iterations. In this run with 10 CMA-ES iterations, the fitness improved but did not reach zero, implying some mismatch remains (with more iterations or different weightings, it could potentially get closer).

Latent Space Analysis: The t-SNE results (and possibly other analysis, like PCA or topological data analysis mentioned in the paper) help visualize the distribution of training scenarios and the position of the true and matched scenarios in latent space. This provides insight into whether the optimizer stayed in a dense region of latent space (indicating realism) and how unique the solution is. For instance, if the true scenario’s latent point lies in a well-populated region and the found solution is nearby, it confirms the method’s ability to recover the correct geological scenario type (one-channel vs two-channel, etc.).

Below is a list of all major dependencies required to run this notebook: 
Notably, torch_geometric (PyTorch Geometric) should be installed with the version compatible to PyTorch (which is implied by torch). 
The custom modules (GVWAE, grid, data, etc.) are part of the project’s codebase and not installable via pip, so they are not listed here. 
The listed requirements cover all external dependencies used in the notebook.
pandas
numpy
tqdm
matplotlib
torch
torchvision
torch_geometric
scipy
scikit-learn
joblib
wandb
