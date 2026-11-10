# Conditional Generative Models for Handwritten CAPTCHAs

This project explores **conditional generative models** to ceate synthetic handwritten digits for CAPTCHA-style human verification

- A **Conditional GAN (cGAN)**
- A **Conditional Diffusion Model**

Both models are trained on **MNIST** and evaluated using:

- **Frechet Inception Distance (FID)** - measures realism & diversity
- **Downstream classifier accuracy** - how useful the synthetic data is for training a classifier

This project is structured with **starter code with ToDos** in both python modules and notebooks.

## Project Structure

```text
project /
|-- data /
    |-- dataloader.py
|-- model /
    |-- __init__.py
    |-- cgan.py
    |-- diffusion.py
|-- training / 
    | -- __init__.py
    | -- train_cgan.py
    | -- train_diffusion.py
|-- utils /
   |-- __init__.py
   |-- checkpoint.py
   |-- metrics.py
   |-- visualize.py
|-- 00_data_preparation.ipynb
|-- 01_cGAN_training.ipynb
|-- 02_diffusion_training.ipynb
|-- 03_evaluation.ipynb
|-- README.md
```

### Data /

- `dataloader.py`
  - Loads the MNIST Dataset
  - Applies standard transform
  - Helper functions to load data in notebooks

### Model /

- `cgan.py`
   **Defines**:
  - **Generator(G)** - takes noise + class label and outputs a 28 x 28 image.
  - **Discriminator(D)** - takes image + class label and outputs real/fake probability

  - **Student ToDos**:
    - Implement the label - conditioning logic in the G and D forward passes.
    - Make sure image tensors are reshaped correctly on output/input

- `diffusion.py`
   **Defines**:
  - `timestep_embedding` - converts diffusion step t into an embedding vector
  - `ResidualBlock` - convolutional block with time + label conditioning and residual connection
  - `ConditionalUNet` - encoder-decoder UNet that predicts noise for a given noisy image, timestep, and label

  - **Student ToDos**:
    - Implement the UNet forward pass:
            time embedding -> down path -> bottleneck -> up path -> output noise prediction

### Training /

- `train_cgan.py`
   Implements the adversarial training loop for the cGAN

  - **Student ToDos**:
    - Complete the Generator update step (Sample noise/labels, generate images, compute generator loss, update weights)
    - Complete the Discriminator update step (compute real vs fake loss, average, backprop)
    - Add Checkpoint saving

- `train_diffusion.py`
   **Defines**:
  - `linear_beta_schedule` - builds a simple noise schedule
  - `sample_images` - runs the reverse diffusion process to sample from noise
  - `train_diffusion` - trains the diffusion model via noise prediction

  - **Student ToDos**:
    - In `train_diffusion`: Implement forward diffusion (add noise) and the MSE noise-prediction training step
    - In `sample_images`: Implement the reverse diffusion update loop to gradually denoise images

### Utils /

`checkpoint.py`: Helper functions to save and load model + optimizer states
`metrics.py`:  Helper functions to calculate evaluation metrics
`visualize.py`: Helper functions for plotting batches of Images and comparison grids

## Notebooks Overview

The noteboks are the main entry point for running the project. They use the modules above and contain additional **notebook-level ToDos** to wire piece together and interpret the results

- `00_data_preparation.ipynb` - Downloads and loads MNIST data, visualizes sample digits
- `01_cGAN_training.ipynb` - Imports `Generator`, `Discriminator` and ``train_cGAN``, sets up models, training loops for specified epocs and visualizes generated digits per class
- `02_diffusion_training.ipynb` - Imports `ConditionalUNet`, `train_diffusion`, and ``sample_images``, trains the difusion model and samples class-conditioned digits
- `03_evaluation.ipynb` - Loads Real MNIST test set, synthetic samples from cGAN and Diffision, runs visual comparison (real vs cGAN vs Diffusion per class), FID computation between real vs synthetic, a simple CNN classifier trained on synthetic data and evaluated on real data

## How to Work with the Starter Code

- Read the docstrings and comments first.
Each module explains its purpose and the role of the ToDos

- Complete the module-level ToDOs before training.
In order:

- `model/cgan.py`
- `model/diffusion.py`
- `training/train_cgan.py`
- `training/train_diffusion.py`

- Then use the notebooks to run experiments and visualize results

## What You Should Aim to Understand

By the end of this project, you should be comfortable with

- How **conditional GANS** and **conditional Diffusion** models generate images from noise + labels
- How to implement basic training loops  for both models in PyTorch
- How to:
  - Visually inspect generative outputs
  - Use **FID** as a realism/diversity metric

Before coding, make sure all libraries are imported correctly, read through the objectives and ToDos, and keep brief notes on what you observe as you train and evaluate both models
