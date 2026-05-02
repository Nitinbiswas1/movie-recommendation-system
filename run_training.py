import kagglehub
import sys
import os
from pathlib import Path

# Add the training directory to the path so we can import the module
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from training.train import MovieRecommenderTrainer

def main():
    print("Downloading TMDB Movies 2023 Dataset...")
    # This downloads the dataset to a local cache directory
    dataset_path = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")
    print(f"Dataset downloaded to: {dataset_path}")
    
    # Initialize trainer using the medium config (recommended)
    trainer = MovieRecommenderTrainer(
        output_dir='./models',
        use_dimensionality_reduction=True,
        n_components=500
    )
    
    # Train model on the top 100K movies (takes ~15 minutes and ~2GB RAM)
    print("\nStarting training... This will take around 15 minutes.")
    df, sim_matrix = trainer.train(
        data_path=dataset_path,
        quality_threshold='medium',
        max_movies=100000
    )

if __name__ == "__main__":
    main()
