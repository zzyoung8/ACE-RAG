import argparse
import json
import os
import random
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from model import ContrastiveEmbedding, ContrastiveLoss, RAGContrastiveDataset

def load_data(data_path):
    """Load the dataset from a JSON file"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data

def train(args):
    # Set seeds for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data_path}")
    data = load_data(args.data_path)
    print(f"Loaded {len(data)} examples")
    
    # Split data into train and validation
    dataset_size = len(data)
    train_size = int(dataset_size * args.train_ratio)
    val_size = dataset_size - train_size
    
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    print(f"Train set: {len(train_data)} examples")
    print(f"Validation set: {len(val_data)} examples")
    
    # Create datasets and dataloaders
    train_dataset = RAGContrastiveDataset(train_data, is_training=True)
    val_dataset = RAGContrastiveDataset(val_data, is_training=True)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ContrastiveEmbedding(
        base_model_name=args.base_model,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim
    ).to(device)
    
    # Initialize loss function and optimizer
    criterion = ContrastiveLoss(temperature=args.temperature)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Training
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_dataloader, desc="Training"):
            optimizer.zero_grad()
            
            queries = batch['query']
            pos_docs = batch['positive_docs']
            neg_docs = batch['negative_docs']
            
            # Forward pass
            query_emb, pos_doc_emb, neg_doc_emb = model(queries, pos_docs, neg_docs)
            
            # Calculate loss
            loss = criterion(query_emb, pos_doc_emb, neg_doc_emb)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Train loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                queries = batch['query']
                pos_docs = batch['positive_docs']
                neg_docs = batch['negative_docs']
                
                # Forward pass
                query_emb, pos_doc_emb, neg_doc_emb = model(queries, pos_docs, neg_docs)
                
                # Calculate loss
                loss = criterion(query_emb, pos_doc_emb, neg_doc_emb)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Validation loss: {avg_val_loss:.4f}")
        
        # Save model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Save the model
            model_path = os.path.join(args.output_dir, f"contrastive_model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, model_path)
            
            print(f"Model saved to {model_path}")
    
    # Save the final model
    final_model_path = os.path.join(args.output_dir, "contrastive_model_final.pt")
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
    }, final_model_path)
    
    print(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a contrastive learning model for RAG")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default="data/zh.json", help="Path to the data file")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data to use for training")
    
    # Model parameters
    parser.add_argument("--base_model", type=str, default="model/paraphrase-multilingual-MiniLM-L12-v2", 
                        help="Base model for sentence embeddings")
    parser.add_argument("--embedding_dim", type=int, default=384, help="Dimension of base embeddings")
    parser.add_argument("--projection_dim", type=int, default=128, help="Dimension of projected embeddings")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for contrastive loss")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="contrastive/saved_models", 
                        help="Directory to save the model")
    
    args = parser.parse_args()
    train(args) 