# Loss Function Design for End-to-End Optical Imaging Systems

This repository contains the PyTorch implementation for jointly training a learnable physical phase mask (DOE) and a digital reconstruction CNN for single-shot HDR imaging. The training pipeline utilizes a Perception-Distortion tradeoff via SphereGAN.

## Environment Setup

It is highly recommended to run this project inside an isolated Python virtual environment to prevent dependency conflicts.

1. **Create the virtual environment:**
   ```bash
   python -m venv venv
   ```
2. **Activate the environment:**
   * **Linux/macOS:** `source venv/bin/activate`
   * **Windows:** `venv\Scripts ctivate`
3. **Install dependencies:**
   Ensure your environment is active, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

The training script expects your HDR dataset to be located at the path defined by `cfg.data_root` in your `config.py` file. 

1. Create the dataset directory (e.g., `data/hdr_images/`).
2. Place your raw HDR image files into this directory.
3. Verify that `config.py` points to this exact location so the `HDRDataset` class can load the files successfully.

## Training Pipeline

Because adversarial gradients can easily destroy the physical optics simulation if introduced too early, the training is split into a **Base Run** (pure L2 structural learning) and a **Branch Run** (SphereGAN perceptual tuning). 

### 1. The Base Run (L2 Optimization)
First, you must train a baseline model from scratch. This allows the `LearnableLens` to form a stable physical Point Spread Function (PSF) and the CNN to learn the coarse image geometry using an L2 loss. By default, the branch script expects a base model trained with a GAN weight of `0.0`.

Run the following command to start the base training:
```bash
python train_sphereGAN.py --base_run --lambda_gan 0.0
```
* This creates a fresh training session starting from Epoch 0.
* Initial, untrained PSF and height map visualizations will be saved to your checkpoint directory automatically for reference.
* Upon completion, the base model is saved to the path defined in `cfg.base_model_path`.

### 2. The Branch Run (Perception-Distortion Trade-off)
Once the base model has stabilized and finished training, you can launch adversarial branches to explore the perception-distortion trade-off. 

The script will automatically detect the saved base model (`lambda_gan=0.0`), load its weights, reset the optimizers, and introduce the SphereGAN discriminator starting from Epoch 0 of the branch.

To run a branch with a specific GAN weight (e.g., `0.01`):
```bash
python train_sphereGAN.py --lambda_gan 0.01
```
You can run this command multiple times with different `--lambda_gan` values (e.g., `1e-4`, `5e-3`, `1e-2`) to generate the data points for your Pareto frontier. 

## Automatic Resumption and Checkpoints

The script includes an automatic resume feature to protect against crashes or timeouts. 

## References

* Deep Optics for Single-Shot High-Dynamic-Range Imaging: https://arxiv.org/abs/1908.00620
* The Perception-Distortion Tradeoff: https://arxiv.org/abs/1711.06077
* Sphere Generative Adversarial Network Based on Geometric Moment Matching: https://arxiv.org/abs/1711.06077


* Checkpoints are saved every 4 epochs into the directory specified by `cfg.check_point_dir`.
* If a training run is interrupted, simply re-run the exact same command you used to start it.
* The script will automatically scan the checkpoint directory, find the latest `ckpt_epoch_*.pt` file, restore the model, lens, optimizer, and scheduler states, and seamlessly resume training from where it left off.
