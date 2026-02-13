"""
Country clustering based on voting patterns.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings


class CountryClusterer:
    """
    Cluster countries into alliances based on voting patterns.
    """
    
    def __init__(self):
        self.vote_matrix = None
        self.countries = None
        self.cluster_labels = None
        self.similarity_matrix = None
    
    def compute_similarity_matrix(
        self,
        vote_matrix: np.ndarray,
        metric: str = 'cosine'
    ) -> np.ndarray:
        """
        Compute similarity matrix between countries.
        
        Args:
            vote_matrix: Country x Issue voting matrix
            metric: Similarity metric ('cosine', 'correlation', 'euclidean')
            
        Returns:
            Similarity matrix
        """
        n_countries = vote_matrix.shape[0]
        similarity_matrix = np.zeros((n_countries, n_countries))
        
        for i in range(n_countries):
            for j in range(n_countries):
                if metric == 'cosine':
                    # Cosine similarity (1 - cosine distance)
                    similarity = 1 - cosine(vote_matrix[i], vote_matrix[j])
                elif metric == 'correlation':
                    # Pearson correlation
                    similarity = np.corrcoef(vote_matrix[i], vote_matrix[j])[0, 1]
                elif metric == 'euclidean':
                    # Negative euclidean distance (normalized)
                    dist = np.linalg.norm(vote_matrix[i] - vote_matrix[j])
                    similarity = -dist / np.sqrt(vote_matrix.shape[1])
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                # Handle NaN values
                if np.isnan(similarity):
                    similarity = 0.0
                
                similarity_matrix[i, j] = similarity
        
        self.similarity_matrix = similarity_matrix
        return similarity_matrix
    
    def cluster_countries(
        self,
        vote_matrix: np.ndarray,
        countries: List[str],
        n_clusters: Optional[int] = None,
        method: str = 'hierarchical'
    ) -> Tuple[np.ndarray, Dict]:
        """
        Cluster countries based on voting patterns.
        
        Args:
            vote_matrix: Country x Issue voting matrix
            countries: List of country names
            n_clusters: Number of clusters (if None, auto-determine)
            method: Clustering method ('kmeans', 'hierarchical')
            
        Returns:
            Tuple of (cluster labels, cluster info dict)
        """
        self.vote_matrix = vote_matrix
        self.countries = countries
        
        # Compute similarity matrix
        self.compute_similarity_matrix(vote_matrix)
        
        # Auto-determine number of clusters if not provided
        if n_clusters is None:
            n_clusters = self._determine_optimal_clusters(vote_matrix)
        
        # Perform clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            self.cluster_labels = clusterer.fit_predict(vote_matrix)
        elif method == 'hierarchical':
            # Use similarity matrix for hierarchical clustering
            # Convert similarity to distance
            distance_matrix = 1 - self.similarity_matrix
            np.fill_diagonal(distance_matrix, 0)
            
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            self.cluster_labels = clusterer.fit_predict(distance_matrix)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Calculate cluster statistics
        cluster_info = self._calculate_cluster_info()
        
        return self.cluster_labels, cluster_info
    
    def _determine_optimal_clusters(
        self,
        vote_matrix: np.ndarray,
        max_clusters: int = 10
    ) -> int:
        """
        Determine optimal number of clusters using silhouette score.
        
        Args:
            vote_matrix: Country x Issue voting matrix
            max_clusters: Maximum number of clusters to try
            
        Returns:
            Optimal number of clusters
        """
        silhouette_scores = []
        min_clusters = 2
        max_clusters = min(max_clusters, len(vote_matrix) - 1)
        
        for k in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(vote_matrix)
            score = silhouette_score(vote_matrix, labels)
            silhouette_scores.append(score)
        
        # Return k with highest silhouette score
        optimal_k = min_clusters + np.argmax(silhouette_scores)
        return optimal_k
    
    def _calculate_cluster_info(self) -> Dict:
        """
        Calculate information about each cluster (alliance).
        
        Returns:
            Dictionary with cluster information
        """
        if self.cluster_labels is None:
            raise ValueError("No clustering performed yet.")
        
        cluster_info = {}
        unique_labels = np.unique(self.cluster_labels)
        
        for label in unique_labels:
            mask = self.cluster_labels == label
            cluster_countries = [c for c, m in zip(self.countries, mask) if m]
            cluster_size = np.sum(mask)
            
            # Calculate average voting pattern for the cluster
            cluster_votes = self.vote_matrix[mask]
            avg_votes = np.mean(cluster_votes, axis=0)
            
            cluster_info[int(label)] = {
                'size': int(cluster_size),
                'countries': cluster_countries,
                'percentage': float(cluster_size / len(self.countries) * 100),
                'avg_voting_pattern': avg_votes.tolist()
            }
        
        return cluster_info
    
    def get_country_alliance(self, country: str) -> int:
        """
        Get alliance (cluster) for a specific country.
        
        Args:
            country: Country name
            
        Returns:
            Alliance label
        """
        if self.cluster_labels is None:
            raise ValueError("No clustering performed yet.")
        
        if country not in self.countries:
            raise ValueError(f"Country '{country}' not found.")
        
        country_idx = self.countries.index(country)
        return int(self.cluster_labels[country_idx])
    
    def get_similar_countries(
        self,
        country: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get most similar countries to a given country.
        
        Args:
            country: Country name
            top_k: Number of similar countries to return
            
        Returns:
            List of (country, similarity) tuples
        """
        if self.similarity_matrix is None:
            raise ValueError("Similarity matrix not computed yet.")
        
        if country not in self.countries:
            raise ValueError(f"Country '{country}' not found.")
        
        country_idx = self.countries.index(country)
        similarities = self.similarity_matrix[country_idx]
        
        # Get indices of top-k most similar countries (excluding itself)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        similar_countries = [
            (self.countries[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return similar_countries
