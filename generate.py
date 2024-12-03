from inference.model_setup import ModelSetup
from inference.angle_utils import AngleProcessor
from inference.sampler import ConformationSampler
import pandas as pd
import pickle
from KDE.KDE2D import KDE2D

def main():
    # Initialize components
    model_setup = ModelSetup(
        tokenizer_path="./tokenizer/",
        model_path="./inference/pretrained_model/"
    )
    angle_processor = AngleProcessor()
    sampler = ConformationSampler(model_setup, angle_processor)

    # Load training data
    train_data = pd.read_csv(
        "./dataset/train.txt", 
        sep=" ", 
        names=list(range(10))
    )

    with open('KDE/kde_functions_from_md.pkl', 'rb') as f:
        kdes = pickle.load(f)

    # Run sampling
    fasta = "GYDPETGTWG"
    accepted_conformations = sampler.sample_conformations(
        train_data,
        num_samples=10000,
        kdes=kdes,  # Assuming kdes is loaded
        fasta=fasta,
        output_path='sampling_results.npy'
    )

if __name__ == "__main__":
    main()