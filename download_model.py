import gdown
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "global_only.pth")

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model weights...")
        # Replace with your actual Google Drive file ID
        gdown.download(
            "https://drive.google.com/uc?id=1KvJQ0YKL-I96UJ5zUGLR_Qpd4R0ach5t",
            MODEL_PATH,
            quiet=False
        )
        print("Model downloaded.")

download_model()
