import pandas as pd
import os
import math
import argparse
import random
from tqdm import tqdm

def split_by_city(input_file, output_dir="split_data", num_splits=5, seed=42):
    """
    Split a CSV file by city into multiple files of roughly equal size.
    
    Args:
        input_file: Path to the input CSV file
        output_dir: Directory to save the split files
        num_splits: Number of files to split into
        seed: Random seed for reproducibility
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the CSV file
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Get unique cities
    city_groups = df.groupby("title")
    unique_cities = list(city_groups.groups.keys())
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle cities to ensure diverse distribution across splits
    random.shuffle(unique_cities)
    
    # Calculate cities per split
    cities_per_split = math.ceil(len(unique_cities) / num_splits)
    
    # Split the cities
    city_splits = [unique_cities[i:i + cities_per_split] for i in range(0, len(unique_cities), cities_per_split)]
    
    # Create splits
    for i, cities in enumerate(city_splits):
        # Filter data for the current split
        split_data = df[df["title"].isin(cities)]
        
        # Create output filename
        output_file = os.path.join(output_dir, f"split_{i+1}_of_{len(city_splits)}.csv")
        
        # Save the split
        split_data.to_csv(output_file, index=False)
        
        print(f"Split {i+1}/{len(city_splits)}: {len(cities)} cities, {len(split_data)} rows saved to {output_file}")
    
    print(f"Splitting complete! {len(city_splits)} files created in {output_dir}")

def split_randomly(input_file, output_dir="split_data", num_splits=5, seed=42):
    """
    Split a CSV file randomly into multiple files of equal size.
    
    Args:
        input_file: Path to the input CSV file
        output_dir: Directory to save the split files
        num_splits: Number of files to split into
        seed: Random seed for reproducibility
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the CSV file
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Calculate the size of each split
    split_size = math.ceil(len(df) / num_splits)
    
    # Create the splits
    for i in range(num_splits):
        # Calculate start and end indices
        start_idx = i * split_size
        end_idx = min((i + 1) * split_size, len(df))
        
        # Extract the subset
        split_data = df.iloc[start_idx:end_idx]
        
        # Create output filename
        output_file = os.path.join(output_dir, f"split_{i+1}_of_{num_splits}.csv")
        
        # Save the split
        split_data.to_csv(output_file, index=False)
        
        print(f"Split {i+1}/{num_splits}: {len(split_data)} rows saved to {output_file}")
    
    print(f"Splitting complete! {num_splits} files created in {output_dir}")

def create_train_val_test_split(input_file, output_dir="dataset_splits", train_pct=0.8, val_pct=0.1, test_pct=0.1, seed=42):
    """
    Split a CSV file into training, validation, and test sets.
    
    Args:
        input_file: Path to the input CSV file
        output_dir: Directory to save the split files
        train_pct: Percentage of data for training
        val_pct: Percentage of data for validation
        test_pct: Percentage of data for testing
        seed: Random seed for reproducibility
    """
    # Validate percentages
    assert abs(train_pct + val_pct + test_pct - 1.0) < 0.001, "Percentages must sum to 1"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the CSV file
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Set random seed
    random.seed(seed)
    
    # Get unique cities to ensure we don't split cities across sets
    cities = list(df["title"].unique())
    random.shuffle(cities)
    
    # Calculate split sizes
    train_size = int(len(cities) * train_pct)
    val_size = int(len(cities) * val_pct)
    
    # Split cities
    train_cities = cities[:train_size]
    val_cities = cities[train_size:train_size + val_size]
    test_cities = cities[train_size + val_size:]
    
    # Create the splits
    train_df = df[df["title"].isin(train_cities)]
    val_df = df[df["title"].isin(val_cities)]
    test_df = df[df["title"].isin(test_cities)]
    
    # Save the splits
    train_file = os.path.join(output_dir, "train.csv")
    val_file = os.path.join(output_dir, "val.csv")
    test_file = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"Training set: {len(train_cities)} cities, {len(train_df)} rows saved to {train_file}")
    print(f"Validation set: {len(val_cities)} cities, {len(val_df)} rows saved to {val_file}")
    print(f"Test set: {len(test_cities)} cities, {len(test_df)} rows saved to {test_file}")
    
    print(f"Dataset splitting complete!")

def main():
    parser = argparse.ArgumentParser(description="Split a dataset into multiple files")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("--output-dir", default="split_data", help="Directory to save the split files")
    parser.add_argument("--mode", choices=["city", "random", "train-val-test"], default="city", 
                        help="Splitting mode: by city, randomly, or into train/val/test")
    parser.add_argument("--num-splits", type=int, default=5, help="Number of files to split into (for city/random modes)")
    parser.add_argument("--train-pct", type=float, default=0.8, help="Percentage for training set (train-val-test mode)")
    parser.add_argument("--val-pct", type=float, default=0.1, help="Percentage for validation set (train-val-test mode)")
    parser.add_argument("--test-pct", type=float, default=0.1, help="Percentage for test set (train-val-test mode)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    if args.mode == "city":
        split_by_city(args.input_file, args.output_dir, args.num_splits, args.seed)
    elif args.mode == "random":
        split_randomly(args.input_file, args.output_dir, args.num_splits, args.seed)
    elif args.mode == "train-val-test":
        create_train_val_test_split(args.input_file, args.output_dir, args.train_pct, args.val_pct, args.test_pct, args.seed)

if __name__ == "__main__":
    main()