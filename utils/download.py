import gdown
import os

# Ensure the directory exists
os.makedirs("pretrained_ckpt", exist_ok=True)

# Download Hifi-GAN vocoder checkpoint
vocoder_checkpoint = "pretrained_ckpt/g_00750000"
vocoder_file_id = "17AI8QlyPbtVUPxT2eozeJrDfw3fcaq7X"
vocoder_url = f"https://drive.google.com/uc?id={vocoder_file_id}"
gdown.download(vocoder_url, vocoder_checkpoint, quiet=False)

# Download kmeans checkpoint
kmeans_checkpoint = "pretrained_ckpt/km.bin"
kmeans_file_id = "1vHSvVdfYXcD8EdnRs3iWfs4LzfOfR2qT"
kmeans_url = f"https://drive.google.com/uc?id={kmeans_file_id}"
gdown.download(kmeans_url, kmeans_checkpoint, quiet=False)

print("Files downloaded successfully!")
