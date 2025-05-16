import argparse
import os
import json
import subprocess

def run_command(command):
    """Run a shell command and print its output"""
    print(f"Running command: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if stdout:
        print(stdout.decode())
    if stderr:
        print(stderr.decode())
    
    return process.returncode

def main(args):
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Train the contrastive learning model
    if args.train:
        print("\n=== Training contrastive learning model ===\n")
        
        train_command = f"python contrastive/train.py \
            --data_path {args.data_path} \
            --train_ratio {args.train_ratio} \
            --base_model {args.base_model} \
            --embedding_dim {args.embedding_dim} \
            --projection_dim {args.projection_dim} \
            --temperature {args.temperature} \
            --num_epochs {args.num_epochs} \
            --batch_size {args.batch_size} \
            --learning_rate {args.learning_rate} \
            --weight_decay {args.weight_decay} \
            --output_dir {args.model_dir}"
        
        run_command(train_command)
        
        print("\n=== Training completed ===\n")
    
    # Step 2: Use the trained model for retrieval
    if args.retrieve:
        print("\n=== Retrieving documents using contrastive learning model ===\n")
        
        # Get the path to the trained model
        model_path = os.path.join(args.model_dir, "contrastive_model_final.pt")
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}. Checking for best epoch model...")
            
            # Try to find the best epoch model
            model_files = [f for f in os.listdir(args.model_dir) if f.startswith("contrastive_model_epoch_")]
            if model_files:
                model_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
                model_path = os.path.join(args.model_dir, model_files[-1])
                print(f"Using model: {model_path}")
            else:
                print("No trained model found. Please train the model first with --train flag.")
                return
        
        # Process the data using the trained model
        retrieval_command = f"python contrastive/retrieval.py \
            --data_path {args.data_path} \
            --model_path {model_path} \
            --output_path {os.path.join(args.output_dir, 'contrastive_retrieved.json')} \
            --base_model {args.base_model} \
            --embedding_dim {args.embedding_dim} \
            --projection_dim {args.projection_dim} \
            --top_k {args.top_k}"
        
        run_command(retrieval_command)
        
        print("\n=== Retrieval completed ===\n")
    
    # Step 3: Run evaluation with the modified data
    if args.evaluate:
        print("\n=== Evaluating with retrieved documents ===\n")
        
        # Check if the retrieved data file exists
        retrieved_data_path = os.path.join(args.output_dir, 'contrastive_retrieved.json')
        if not os.path.exists(retrieved_data_path):
            print(f"Retrieved data file not found at {retrieved_data_path}. Please run retrieval first with --retrieve flag.")
            return
        
        # Run the evaluation using RAG.py
        evaluation_command = f"python RAG.py \
            --dataset {args.dataset} \
            --modelname {args.modelname} \
            --plm {args.plm} \
            --method {args.method} \
            --temperature {args.generation_temperature}"
        
        # Create symbolic link to make the retrieved data accessible as the dataset
        original_data_path = f"data/{args.dataset}.json"
        if os.path.exists(original_data_path):
            # Backup the original data
            backup_command = f"cp {original_data_path} {original_data_path}.backup"
            run_command(backup_command)
            
            # Copy the retrieved data to the dataset location
            copy_command = f"cp {retrieved_data_path} {original_data_path}"
            run_command(copy_command)
            
            # Run the evaluation
            run_command(evaluation_command)
            
            # Restore the original data
            restore_command = f"mv {original_data_path}.backup {original_data_path}"
            run_command(restore_command)
        else:
            print(f"Original dataset file not found at {original_data_path}. Cannot run evaluation.")
            return
        
        print("\n=== Evaluation completed ===\n")
        
        # Print the results
        result_file = f'results/{args.dataset}/{args.modelname}_{args.method}_metrics.json'
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                print("\n=== Evaluation Results ===")
                print(f"Accuracy: {results['accuracy']:.4f}")
                print(f"Match Score: {results['match_avg_score']:.4f}")
                print(f"Precision: {results['precision_avg_score']:.4f}")
                print(f"Recall: {results['recall_avg_score']:.4f}")
                print(f"F1 Score: {results['f1_avg_score']:.4f}")
        else:
            print(f"Result file not found at {result_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run contrastive learning pipeline for RAG")
    
    # Pipeline control
    parser.add_argument("--train", action="store_true", help="Train the contrastive learning model")
    parser.add_argument("--retrieve", action="store_true", help="Use the trained model for retrieval")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate using the retrieved documents")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default="data/zh.json", help="Path to the data file")
    parser.add_argument("--dataset", type=str, default="zh", help="Dataset name for evaluation")
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
    
    # Retrieval parameters
    parser.add_argument("--top_k", type=int, default=15, help="Number of documents to retrieve")
    
    # Evaluation parameters
    parser.add_argument("--modelname", type=str, default="Qwen2.5", help="Model name for evaluation")
    parser.add_argument("--plm", type=str, default="./model/Qwen2.5-7B-Instruct", help="Path to the model")
    parser.add_argument("--method", type=str, default="zzy", help="Method for evaluation")
    parser.add_argument("--generation_temperature", type=float, default=0.2, help="Temperature for generation")
    
    # Output parameters
    parser.add_argument("--model_dir", type=str, default="contrastive/saved_models", 
                        help="Directory to save the model")
    parser.add_argument("--output_dir", type=str, default="contrastive/output", 
                        help="Directory to save the output")
    
    args = parser.parse_args()
    
    # If no flags are provided, run the full pipeline
    if not (args.train or args.retrieve or args.evaluate):
        args.train = True
        args.retrieve = True
        args.evaluate = True
    
    main(args) 