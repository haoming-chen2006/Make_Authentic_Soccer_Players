# Make Authentic Soccer Players

This project uses **various simple machine learning models** to generate soccer player profiles - this includes images, names, statistics, and background information. The focus is to provide accurate generaiton based on a small dataset. 

List of models:
  GAN
    - Archeitecture following the FASTGAN approach
  Vector Quantized Variational Autoencoder (VQ-VAE)
    - The architecture is based on convolutional encoders/decoders and vector quantization, following the design in *Neural Discrete Representation Learning* (van den Oord et al., 2017).
  Transformer based Makemore
    - Generate player names
  GPT 2 Model
    - Generate player background story.
  

---

## Results

Generated images


Generated profile with names


Generated full profile

---

## 🏃‍♂️ Training

```bash
python train.py train_vqvae_cnn

