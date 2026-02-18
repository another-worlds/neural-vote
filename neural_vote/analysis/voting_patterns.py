"""
Analysis of voting patterns, interrelations, and metrics.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.stats import chi2_contingency, pearsonr
from collections import defaultdict


class VotingAnalyzer:
    """
    Analyze voting patterns, interrelations, and calculate importance/conflict metrics.
    """
    
    def __init__(self, voting_data: pd.DataFrame, vote_matrix: np.ndarray):
        """
        Initialize analyzer.
        
        Args:
            voting_data: DataFrame with voting data
            vote_matrix: Country x Issue voting matrix
        """
        self.voting_data = voting_data
        self.vote_matrix = vote_matrix
        self.countries = sorted(voting_data['country'].unique().tolist())
        self.issues = sorted(voting_data['issue_id'].unique().tolist())
    
    def calculate_issue_importance(self) -> Dict[str, float]:
        """
        Calculate importance score for each issue based on voting patterns.
        
        Importance is measured by:
        - Voting participation rate
        - Variance in votes (controversial issues are important)
        - Number of countries voting
        
        Returns:
            Dictionary mapping issue_id to importance score
        """
        importance_scores = {}
        
        for issue_id in self.issues:
            issue_votes = self.voting_data[
                self.voting_data['issue_id'] == issue_id
            ]['vote']
            
            # Participation rate
            participation = len(issue_votes) / len(self.countries)
            
            # Variance in votes (higher variance = more controversial)
            variance = np.var(issue_votes)
            
            # Normalized importance score
            importance = (participation * 0.5 + variance * 0.5)
            
            importance_scores[issue_id] = float(importance)
        
        return importance_scores
    
    def calculate_conflict_matrix(self) -> np.ndarray:
        """
        Calculate conflict matrix between countries.
        
        Conflict is measured by the disagreement in voting patterns.
        
        Returns:
            Country x Country conflict matrix
        """
        n_countries = len(self.countries)
        conflict_matrix = np.zeros((n_countries, n_countries))
        
        for i in range(n_countries):
            for j in range(n_countries):
                # Calculate voting disagreement
                votes_i = self.vote_matrix[i]
                votes_j = self.vote_matrix[j]
                
                # Count disagreements (opposite votes)
                disagreements = np.sum(
                    (votes_i == 1) & (votes_j == -1) |
                    (votes_i == -1) & (votes_j == 1)
                )
                
                # Normalize by total votes
                total_votes = len(votes_i)
                conflict = disagreements / total_votes if total_votes > 0 else 0
                
                conflict_matrix[i, j] = conflict
        
        return conflict_matrix
    
    def calculate_agreement_matrix(self) -> np.ndarray:
        """
        Calculate agreement matrix between countries.
        
        Returns:
            Country x Country agreement matrix
        """
        n_countries = len(self.countries)
        agreement_matrix = np.zeros((n_countries, n_countries))
        
        for i in range(n_countries):
            for j in range(n_countries):
                votes_i = self.vote_matrix[i]
                votes_j = self.vote_matrix[j]
                
                # Count agreements (same votes)
                agreements = np.sum(votes_i == votes_j)
                
                # Normalize by total votes
                total_votes = len(votes_i)
                agreement = agreements / total_votes if total_votes > 0 else 0
                
                agreement_matrix[i, j] = agreement
        
        return agreement_matrix
    
    def analyze_issue_correlations(self) -> pd.DataFrame:
        """
        Analyze correlations between different issues.
        
        Returns:
            DataFrame with issue pairs and their correlation
        """
        correlations = []
        
        # Get vote matrix transposed (issues x countries)
        issue_matrix = self.vote_matrix.T
        
        for i, issue_i in enumerate(self.issues):
            for j, issue_j in enumerate(self.issues):
                if i < j:  # Only upper triangle
                    # Calculate correlation
                    corr, p_value = pearsonr(issue_matrix[i], issue_matrix[j])
                    
                    if not np.isnan(corr):
                        correlations.append({
                            'issue_1': issue_i,
                            'issue_2': issue_j,
                            'correlation': corr,
                            'p_value': p_value
                        })
        
        return pd.DataFrame(correlations)
    
    def analyze_cluster_voting_patterns(
        self,
        cluster_labels: np.ndarray,
        cluster_type: str = 'issue'
    ) -> Dict:
        """
        Analyze voting patterns within and across clusters.
        
        Args:
            cluster_labels: Cluster labels for issues or countries
            cluster_type: 'issue' or 'country'
            
        Returns:
            Dictionary with cluster voting statistics
        """
        cluster_stats = defaultdict(dict)
        unique_clusters = np.unique(cluster_labels)
        
        if cluster_type == 'issue':
            # Analyze how countries vote on different issue clusters
            for cluster in unique_clusters:
                cluster_mask = cluster_labels == cluster
                cluster_issues_indices = np.where(cluster_mask)[0]
                
                # Get votes for issues in this cluster
                cluster_votes = self.vote_matrix[:, cluster_issues_indices]
                
                # Calculate statistics
                avg_vote = np.mean(cluster_votes)
                std_vote = np.std(cluster_votes)
                
                # Calculate yes/no/abstain percentages
                yes_pct = np.sum(cluster_votes == 1) / cluster_votes.size * 100
                no_pct = np.sum(cluster_votes == -1) / cluster_votes.size * 100
                abstain_pct = np.sum(cluster_votes == 0) / cluster_votes.size * 100
                
                cluster_stats[int(cluster)] = {
                    'avg_vote': float(avg_vote),
                    'std_vote': float(std_vote),
                    'yes_percentage': float(yes_pct),
                    'no_percentage': float(no_pct),
                    'abstain_percentage': float(abstain_pct),
                    'n_issues': int(np.sum(cluster_mask))
                }
        
        elif cluster_type == 'country':
            # Analyze voting cohesion within country clusters
            for cluster in unique_clusters:
                cluster_mask = cluster_labels == cluster
                cluster_countries_indices = np.where(cluster_mask)[0]
                
                # Get votes for countries in this cluster
                cluster_votes = self.vote_matrix[cluster_countries_indices]
                
                # Calculate intra-cluster agreement
                agreements = []
                for i in range(len(cluster_countries_indices)):
                    for j in range(i+1, len(cluster_countries_indices)):
                        vote_i = cluster_votes[i]
                        vote_j = cluster_votes[j]
                        agreement = np.sum(vote_i == vote_j) / len(vote_i)
                        agreements.append(agreement)
                
                avg_agreement = np.mean(agreements) if agreements else 0
                
                cluster_stats[int(cluster)] = {
                    'avg_agreement': float(avg_agreement),
                    'n_countries': int(np.sum(cluster_mask))
                }
        
        return dict(cluster_stats)
    
    def get_most_controversial_issues(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get the most controversial issues.
        
        Args:
            top_k: Number of issues to return
            
        Returns:
            List of (issue_id, controversy_score) tuples
        """
        controversy_scores = {}
        
        for i, issue_id in enumerate(self.issues):
            issue_votes = self.vote_matrix[:, i]
            
            # Controversy = variance in votes + balance between yes/no
            variance = np.var(issue_votes)
            
            yes_count = np.sum(issue_votes == 1)
            no_count = np.sum(issue_votes == -1)
            total = yes_count + no_count
            
            # Balance score (closer to 0.5 = more controversial)
            if total > 0:
                balance = 1 - abs(yes_count / total - 0.5) * 2
            else:
                balance = 0
            
            controversy = variance * balance
            controversy_scores[issue_id] = controversy
        
        # Sort by controversy score
        sorted_issues = sorted(
            controversy_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_issues[:top_k]
    
    def calculate_country_influence(self) -> Dict[str, float]:
        """
        Calculate influence score for each country.
        
        Influence is based on:
        - Voting participation
        - Agreement with final outcomes
        - Centrality in the voting network
        
        Returns:
            Dictionary mapping country to influence score
        """
        influence_scores = {}
        
        for i, country in enumerate(self.countries):
            country_votes = self.vote_matrix[i]
            
            # Participation rate (non-abstain votes)
            participation = np.sum(country_votes != 0) / len(country_votes)
            
            # Agreement with majority
            issue_majorities = np.sign(np.sum(self.vote_matrix, axis=0))
            agreement_with_majority = np.sum(
                country_votes == issue_majorities
            ) / len(country_votes)
            
            # Combined influence score
            influence = (participation * 0.4 + agreement_with_majority * 0.6)
            
            influence_scores[country] = float(influence)
        
        return influence_scores
