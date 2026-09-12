# Loss Function Design for End-to-End Optical Imaging Systems

This repository contains the PyTorch implementation for jointly training a learnable physical phase mask (DOE) and a digital reconstruction CNN for single-shot HDR imaging. The training pipeline utilizes a Perception-Distortion tradeoff via SphereGAN.

## Environment Setup

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

The training HDR dataset should be placed at the path defined by `data_root` inside `config.py` file.
We have used the same dataset as used in reference [1]. See instructions there: https://github.com/computational-imaging/DeepOpticsHDR

Then run `preprocess_training_data.sh`, after updating the paths.

## Training Pipeline

Because adversarial gradients are notoriously unstable, in addition to training with GAN loss from the beginning, we allow splitting the training into a **Base Run** (pure L2 structural learning) and a **Branch Run** (SphereGAN perceptual tuning). 

### 1. The Base Run

Run the following command to start the base training:
```bash
python train_sphereGAN.py --base_run --lambda_gan 0.0
```
* You may choose a different value for `lambda_gan` (weight of the GAN loss relative to the L2 loss). However training a "branched network" always uses base model with `lambda_gan = 0`
* Upon completion, the base model is saved to the path defined in `base_model_path` inside `config.py`.

### 2. The Branch Run
Once you have a trained base network, you can tune it by adding adversarial loss.

The script will automatically detect the saved base model (`lambda_gan=0.0`), load its weights, reset the optimizers, and introduce the SphereGAN discriminator starting from Epoch 0 of the branch.

To run a branch with a specific GAN weight (e.g., `0.01`):
```bash
python train_sphereGAN.py --lambda_gan 0.01
```
You can run this command multiple times with different `--lambda_gan` values (e.g., `1e-4`, `5e-3`, `1e-2`).

## Automatic Resumption and Checkpoints

The script includes an automatic resume feature to protect against crashes or timeouts. 

* Checkpoints are saved every 4 epochs into the directory specified by `cfg.check_point_dir`.
* If a training run is interrupted, simply re-run the exact same command you used to start it.
* The script will automatically scan the checkpoint directory, find the latest `ckpt_epoch_*.pt` file, restore the model, lens, optimizer, and scheduler states, and seamlessly resume training from where it left off.

## References

* Deep Optics for Single-Shot High-Dynamic-Range Imaging: https://arxiv.org/abs/1908.00620
* The Perception-Distortion Tradeoff: https://arxiv.org/abs/1711.06077
* Sphere Generative Adversarial Network Based on Geometric Moment Matching: https://arxiv.org/abs/1711.06077
