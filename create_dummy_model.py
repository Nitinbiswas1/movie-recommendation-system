import pandas as pd
import numpy as np
import json
from pathlib import Path

def create_demo_model():
    output_dir = Path('./static')
    output_dir.mkdir(exist_ok=True)
    
    # Create minimal movie data
    movies = [
        {
            'id': 1,
            'title': 'The Matrix',
            'release_date': '1999-03-31',
            'primary_company': 'Warner Bros.',
            'genres': ['Action', 'Sci-Fi'],
            'vote_average': 8.7,
            'vote_count': 20000,
            'popularity': 100.0,
            'overview': 'A computer hacker learns from mysterious rebels about the true nature of his reality.',
            'imdb_id': 'tt0133093',
            'poster_path': '/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg'
        },
        {
            'id': 2,
            'title': 'The Matrix Reloaded',
            'release_date': '2003-05-15',
            'primary_company': 'Warner Bros.',
            'genres': ['Action', 'Sci-Fi'],
            'vote_average': 7.2,
            'vote_count': 10000,
            'popularity': 80.0,
            'overview': 'Freedom fighters Neo, Trinity and Morpheus continue to lead the revolt against the Machine Army.',
            'imdb_id': 'tt0234215',
            'poster_path': '/aA5qHS045KKtoEAAi2bB2qlfA1B.jpg'
        },
        {
            'id': 3,
            'title': 'Inception',
            'release_date': '2010-07-16',
            'primary_company': 'Warner Bros.',
            'genres': ['Action', 'Sci-Fi', 'Thriller'],
            'vote_average': 8.8,
            'vote_count': 30000,
            'popularity': 120.0,
            'overview': 'A thief who steals corporate secrets through the use of dream-sharing technology.',
            'imdb_id': 'tt1375666',
            'poster_path': '/8Z8dptjXlshT677nB2w0HkofjT.jpg'
        }
    ]
    
    df = pd.DataFrame(movies)
    
    # Save parquet
    df.to_parquet(output_dir / 'movie_metadata.parquet', compression='gzip', index=True)
    
    # Create similarity matrix (3x3)
    # Matrix: Matrix Reloaded is very similar, Inception is somewhat similar
    similarity_matrix = np.array([
        [1.0, 0.9, 0.6],
        [0.9, 1.0, 0.5],
        [0.6, 0.5, 1.0]
    ], dtype=np.float32)
    
    np.save(output_dir / 'similarity_matrix.npy', similarity_matrix)
    
    # Save title to index
    title_to_idx = {row['title']: idx for idx, row in df.iterrows()}
    with open(output_dir / 'title_to_idx.json', 'w') as f:
        json.dump(title_to_idx, f)
        
    # Save config
    config = {
        'n_movies': len(movies),
        'use_svd': False,
        'n_components': None,
        'matrix_shape': similarity_matrix.shape,
        'dataset': 'Demo Dataset (3 movies)'
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    create_demo_model()
    print("Demo model created in static/")
