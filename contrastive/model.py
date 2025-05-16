import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader

class ContrastiveEmbedding(nn.Module):
    """
    Contrastive learning model for document and query embeddings
    """
    def __init__(self, base_model_name='model/paraphrase-multilingual-MiniLM-L12-v2', embedding_dim=384, projection_dim=128):
        super(ContrastiveEmbedding, self).__init__()
        
        # Load base embedding model (SentenceTransformer)
        self.base_model = SentenceTransformer(base_model_name)
        self.embedding_dim = embedding_dim
        
        # Projection layers to transform embeddings to a space where contrastive loss is applied
        self.query_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        self.doc_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
    def encode_query(self, queries, convert_to_tensor=True):
        """Encode queries and project them to the contrastive space"""
        # Get base embeddings
        query_embeddings = self.base_model.encode(queries, convert_to_tensor=convert_to_tensor)
        if convert_to_tensor and not isinstance(query_embeddings, torch.Tensor):
            query_embeddings = torch.tensor(query_embeddings)
        
        # Project to contrastive space
        return self.query_projection(query_embeddings)
    
    def encode_document(self, documents, convert_to_tensor=True):
        """Encode documents and project them to the contrastive space"""
        # Get base embeddings
        doc_embeddings = self.base_model.encode(documents, convert_to_tensor=convert_to_tensor)
        if convert_to_tensor and not isinstance(doc_embeddings, torch.Tensor):
            doc_embeddings = torch.tensor(doc_embeddings)
        
        # Project to contrastive space
        return self.doc_projection(doc_embeddings)
    
    def forward(self, queries, positive_docs, negative_docs=None):
        """
        Forward pass for contrastive learning
        
        Args:
            queries: List of query texts
            positive_docs: List of positive document texts (relevant to queries)
            negative_docs: List of negative document texts (optional)
            
        Returns:
            query_embeddings: Projected query embeddings
            pos_doc_embeddings: Projected positive document embeddings
            neg_doc_embeddings: Projected negative document embeddings (if provided)
        """
        # Get query and document embeddings
        query_embeddings = self.encode_query(queries)
        pos_doc_embeddings = self.encode_document(positive_docs)
        
        if negative_docs is not None:
            neg_doc_embeddings = self.encode_document(negative_docs)
            return query_embeddings, pos_doc_embeddings, neg_doc_embeddings
        
        return query_embeddings, pos_doc_embeddings

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training the model
    """
    def __init__(self, temperature=0.07, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, query_embeddings, pos_doc_embeddings, neg_doc_embeddings=None):
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        pos_doc_embeddings = F.normalize(pos_doc_embeddings, p=2, dim=1)
        
        # Calculate similarity between query and positive documents
        pos_similarity = torch.sum(query_embeddings * pos_doc_embeddings, dim=1)
        
        if neg_doc_embeddings is not None:
            # Use provided negative documents
            neg_doc_embeddings = F.normalize(neg_doc_embeddings, p=2, dim=1)
            neg_similarity = torch.sum(query_embeddings * neg_doc_embeddings, dim=1)
            
            # Calculate InfoNCE loss with positive and negative pairs
            loss = -torch.log(
                torch.exp(pos_similarity / self.temperature) / 
                (torch.exp(pos_similarity / self.temperature) + 
                 torch.exp(neg_similarity / self.temperature))
            ).mean()
        else:
            # Use in-batch negatives
            batch_size = query_embeddings.size(0)
            
            # Calculate similarity matrix for all query-document pairs in batch
            similarity_matrix = torch.matmul(query_embeddings, pos_doc_embeddings.T) / self.temperature
            
            # Mask for positive pairs (diagonal elements)
            mask = torch.eye(batch_size, device=query_embeddings.device)
            
            # Calculate InfoNCE loss with in-batch negatives
            exp_sim = torch.exp(similarity_matrix)
            log_prob = torch.log(exp_sim * mask / (exp_sim.sum(dim=1, keepdim=True) + 1e-8))
            
            loss = -log_prob.sum(dim=1).mean()
            
        return loss

class RAGContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning with RAG data
    """
    def __init__(self, data, is_training=True):
        self.data = data
        self.is_training = is_training
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['question']
        
        # For training: use doc as positive and randomly select negatives
        if self.is_training:
            # Get positive document (document containing the answer)
            docs = item['doc']
            answer = item['answer'][0] if isinstance(item['answer'], list) else item['answer']
            
            pos_docs = []
            neg_docs = []
            
            for doc in docs:
                if answer in doc:
                    pos_docs.append(doc)
                else:
                    neg_docs.append(doc)
            
            # If no positive documents found, use the first document as positive
            if len(pos_docs) == 0:
                pos_docs.append(docs[0])
            
            # If no negative documents found, create synthetic ones
            if len(neg_docs) == 0 and len(self.data) > 1:
                # Use documents from other examples as negatives
                random_idx = (idx + 1) % len(self.data)
                neg_docs = self.data[random_idx]['doc']
            
            return {
                'query': query,
                'positive_docs': pos_docs[0] if pos_docs else "",
                'negative_docs': neg_docs[0] if neg_docs else "",
                'answer': answer
            }
        else:
            # For evaluation: just return the query and documents
            return {
                'query': query,
                'docs': item['doc'],
                'answer': item['answer'][0] if isinstance(item['answer'], list) else item['answer']
            } 