import argparse
import json
import os
import torch
# import faiss  # 移除faiss依赖
import numpy as np
from tqdm import tqdm
from model import ContrastiveEmbedding
from sklearn.neighbors import NearestNeighbors  # 使用scikit-learn替代

class ContrastiveRetriever:
    """
    Retriever that uses contrastive learning model to encode queries and documents
    and retrieve relevant documents for a given query
    """
    def __init__(self, model_path, base_model_name='model/paraphrase-multilingual-MiniLM-L12-v2', 
                 embedding_dim=384, projection_dim=128, use_gpu=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        
        # Initialize the model
        self.model = ContrastiveEmbedding(
            base_model_name=base_model_name,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim
        ).to(self.device)
        
        # Load the trained model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Dimensions for index
        self.projection_dim = projection_dim
        
        # Initialize index
        self.index = None
        self.documents = []
        self.embeddings = None
    
    def build_index(self, data, batch_size=32):
        """
        Build nearest neighbors index for fast document retrieval
        
        Args:
            data: List of data items, each containing 'doc' field
            batch_size: Batch size for encoding documents
        """
        # Extract documents from data
        all_documents = []
        for item in data:
            all_documents.extend(item['doc'])
        
        # Remove duplicates while preserving order
        unique_documents = []
        seen = set()
        for doc in all_documents:
            if doc not in seen:
                seen.add(doc)
                unique_documents.append(doc)
        
        # Store the documents for later retrieval
        self.documents = unique_documents
        print(f"Building index with {len(self.documents)} documents")
        
        # Encode documents
        document_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.documents), batch_size), desc="Encoding documents"):
                batch_docs = self.documents[i:i+batch_size]
                batch_embeddings = self.model.encode_document(batch_docs).cpu().numpy()
                document_embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(document_embeddings)
        self.embeddings = all_embeddings
        
        # Build nearest neighbors index using cosine similarity
        self.index = NearestNeighbors(n_neighbors=min(len(self.documents), 100), 
                                      metric='cosine', 
                                      algorithm='brute')
        self.index.fit(all_embeddings)
        
        print(f"Index built with {len(self.documents)} documents")
    
    def retrieve(self, query, top_k=15):
        """
        Retrieve the most relevant documents for a query
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        # Encode the query
        with torch.no_grad():
            query_embedding = self.model.encode_query([query]).cpu().numpy()
        
        # Search the index - for NearestNeighbors, smaller distance means more similar
        top_k = min(top_k, len(self.documents))
        distances, indices = self.index.kneighbors(query_embedding, n_neighbors=top_k)
        
        # Sort by smallest distance
        sorted_indices = indices[0]
        
        # Get the documents
        retrieved_documents = [self.documents[idx] for idx in sorted_indices]
        
        return retrieved_documents

def process_data(data_path, model_path, output_path, base_model, embedding_dim, projection_dim, top_k):
    """
    Process data with contrastive retrieval and save results
    
    Args:
        data_path: Path to the input data
        model_path: Path to the trained model
        output_path: Path to save the output data
        base_model: Base model name
        embedding_dim: Dimension of base embeddings
        projection_dim: Dimension of projected embeddings
        top_k: Number of documents to retrieve for each query
    """
    # Load data
    print(f"Loading data from {data_path}")
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    
    print(f"Loaded {len(data)} examples")
    
    # Initialize retriever
    retriever = ContrastiveRetriever(
        model_path=model_path,
        base_model_name=base_model,
        embedding_dim=embedding_dim,
        projection_dim=projection_dim
    )
    
    # Build index
    retriever.build_index(data)
    
    # Process each query and retrieve documents
    processed_data = []
    
    for item in tqdm(data, desc="Processing queries"):
        query = item['question']
        retrieved_docs = retriever.retrieve(query, top_k=top_k)
        
        # Make a copy of the original item and update its documents
        new_item = item.copy()
        new_item['doc'] = retrieved_docs
        
        processed_data.append(new_item)
    
    # Save processed data
    print(f"Saving processed data to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data with contrastive retrieval")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output data")
    
    # Model parameters
    parser.add_argument("--base_model", type=str, default="model/paraphrase-multilingual-MiniLM-L12-v2", 
                        help="Base model for sentence embeddings")
    parser.add_argument("--embedding_dim", type=int, default=384, help="Dimension of base embeddings")
    parser.add_argument("--projection_dim", type=int, default=128, help="Dimension of projected embeddings")
    
    # Retrieval parameters
    parser.add_argument("--top_k", type=int, default=15, help="Number of documents to retrieve")
    
    args = parser.parse_args()
    
    process_data(
        data_path=args.data_path,
        model_path=args.model_path,
        output_path=args.output_path,
        base_model=args.base_model,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        top_k=args.top_k
    ) 