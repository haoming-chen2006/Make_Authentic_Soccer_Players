# Make Authentic Soccer Players

This project uses **various machine learning models** to generate soccer player profiles - this includes **images, names, statistics, and background information.** The focus is to provide accurate generaiton based on a small dataset. This project serve as an educational project for me to build models from scratch and understand their archeitecture. With possible integration into any games or applications requiring seamless generation of authentic soccer play profiles

# List of models:
  GAN  
    - Archeitecture following the FASTGAN approach   
    
  Vector Quantized Variational Autoencoder (VQ-VAE)  
      - The architecture is based on convolutional encoders/decoders and vector quantization, following the design in *Neural Discrete Representation Learning* (van den Oord et al., 2017). 
 
 Transformer based Makemore  
   - Generate player names
   - Folllowing approach in the videos of kaparthy
     
GPT 2 Model
  - Built from scratch to tenerate player background story and names, following official approach in pytorch tutorials
  - Include a flash attention and faster multihead attention implementation
      
  Conditional Quantizer:  
  - Basically the VQ_VAE design with Transformer model as encoder and decoders to better learn semantic meaning
  

---

## Results

Generated images

By VQVAE:
<img width="738" alt="Screenshot 2025-05-21 at 14 28 17" src="https://github.com/user-attachments/assets/db273cc2-a8c9-4961-9740-00abf405d524" />

Generated profile with names:
Name: Tah Dunk
Position: Forward
Nationality: French
Club: FC Montclair
Height: 6'1" (185 cm)
Age: 24
Preferred Foot: Right

Biography:
Tah Dunk is a fast and skillful striker known for his powerful shots and flair on the ball. Born in Marseille, he grew up playing street football before joining the FC Montclair academy at age 14. Since making his debut for the first team, Dunk has become a key player, scoring goals with both feet and creating chances with his clever movement. Fans admire his energy on the pitch and his signature backflip celebration. He dreams of playing for the national team and winning trophies in Europe.


---

## 🏃‍♂️ Training

The dataset is avaliable at https://www.kaggle.com/datasets/birdy654/football-players-and-staff-faces

```bash
python train.py [train_vqvae_cnn/train_GAN/train_Transformer]


