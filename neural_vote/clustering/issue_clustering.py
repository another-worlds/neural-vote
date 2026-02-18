"""
Semantic clustering of issues using neural embeddings.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import warnings

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence-transformers not available. Install it for semantic clustering.")


class IssueClusterer:
    """
    Semantic clustering of issues using neural embeddings.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize issue clusterer.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.cluster_labels = None
        self.n_clusters = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer(model_name)
    
    def encode_issues(self, issue_descriptions: List[str]) -> np.ndarray:
        """
        Encode issue descriptions to embeddings.
        
        Args:
            issue_descriptions: List of issue descriptions
            
        Returns:
            Array of embeddings
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            # Fallback to simple TF-IDF if sentence-transformers not available
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=384)
            self.embeddings = vectorizer.fit_transform(issue_descriptions).toarray()
        else:
            self.embeddings = self.model.encode(issue_descriptions, show_progress_bar=True)
        
        return self.embeddings
    
    def cluster_issues(
        self,
        issue_descriptions: List[str],
        n_clusters: Optional[int] = None,
        method: str = 'kmeans'
    ) -> Tuple[np.ndarray, Dict]:
        """
        Cluster issues based on their semantic content.
        
        Args:
            issue_descriptions: List of issue descriptions
            n_clusters: Number of clusters (if None, auto-determine)
            method: Clustering method ('kmeans', 'dbscan', 'hierarchical')
            
        Returns:
            Tuple of (cluster labels, cluster info dict)
        """
        # Encode issues if not already done
        if self.embeddings is None:
            self.encode_issues(issue_descriptions)
        
        # Auto-determine number of clusters if not provided
        if n_clusters is None and method == 'kmeans':
            n_clusters = self._determine_optimal_clusters(self.embeddings)
        
        self.n_clusters = n_clusters
        
        # Perform clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            self.cluster_labels = clusterer.fit_predict(self.embeddings)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=2)
            self.cluster_labels = clusterer.fit_predict(self.embeddings)
        elif method == 'hierarchical':
            if n_clusters is None:
                n_clusters = 5
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            self.cluster_labels = clusterer.fit_predict(self.embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Calculate cluster statistics
        cluster_info = self._calculate_cluster_info(issue_descriptions)
        
        return self.cluster_labels, cluster_info
    
    def _determine_optimal_clusters(
        self,
        embeddings: np.ndarray,
        max_clusters: int = 10
    ) -> int:
        """
        Determine optimal number of clusters using elbow method.
        
        Args:
            embeddings: Issue embeddings
            max_clusters: Maximum number of clusters to try
            
        Returns:
            Optimal number of clusters
        """
        silhouette_scores = []
        min_clusters = 2
        max_clusters = min(max_clusters, len(embeddings) - 1)
        
        for k in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            silhouette_scores.append(score)
        
        # Return k with highest silhouette score
        optimal_k = min_clusters + np.argmax(silhouette_scores)
        return optimal_k
    
    def _calculate_cluster_info(self, issue_descriptions: List[str]) -> Dict:
        """
        Calculate information about each cluster.
        
        Args:
            issue_descriptions: List of issue descriptions
            
        Returns:
            Dictionary with cluster information
        """
        if self.cluster_labels is None:
            raise ValueError("No clustering performed yet.")
        
        cluster_info = {}
        unique_labels = np.unique(self.cluster_labels)
        
        for label in unique_labels:
            mask = self.cluster_labels == label
            cluster_issues = [desc for desc, m in zip(issue_descriptions, mask) if m]
            cluster_size = np.sum(mask)
            
            cluster_info[int(label)] = {
                'size': int(cluster_size),
                'issues': cluster_issues[:5],  # Store first 5 issues as examples
                'percentage': float(cluster_size / len(issue_descriptions) * 100)
            }
        
        return cluster_info
    
    def get_cluster_label(self, issue_index: int) -> int:
        """
        Get cluster label for a specific issue.
        
        Args:
            issue_index: Index of the issue
            
        Returns:
            Cluster label
        """
        if self.cluster_labels is None:
            raise ValueError("No clustering performed yet.")
        
        return int(self.cluster_labels[issue_index])
    
    def assign_votes_to_clusters(
        self,
        voting_data,
        issue_ids: List[str],
        issue_descriptions: List[str]
    ):
        """
        Assign cluster labels to voting data.
        
        Args:
            voting_data: DataFrame with voting data
            issue_ids: List of issue IDs
            issue_descriptions: List of issue descriptions
            
        Returns:
            Updated voting data with cluster labels
        """
        # Cluster issues
        labels, _ = self.cluster_issues(issue_descriptions)
        
        # Create mapping from issue_id to cluster
        issue_to_cluster = dict(zip(issue_ids, labels))
        
        # Add cluster column to voting data
        voting_data['cluster'] = voting_data['issue_id'].map(issue_to_cluster)
        
        return voting_data
