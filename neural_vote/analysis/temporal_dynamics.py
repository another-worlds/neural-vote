"""
Temporal dynamics analysis for voting relationships over time.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict


class TemporalAnalyzer:
    """
    Analyze temporal dynamics of voting relationships.
    """
    
    def __init__(self, voting_data: pd.DataFrame):
        """
        Initialize temporal analyzer.
        
        Args:
            voting_data: DataFrame with voting data (must include 'date' column)
        """
        if 'date' not in voting_data.columns:
            raise ValueError("Voting data must include 'date' column for temporal analysis.")
        
        self.voting_data = voting_data.copy()
        self.voting_data['date'] = pd.to_datetime(self.voting_data['date'])
        self.countries = sorted(voting_data['country'].unique().tolist())
        self.issues = sorted(voting_data['issue_id'].unique().tolist())
    
    def create_time_windows(
        self,
        window_size: str = '1Y'
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Create time windows for analysis.
        
        Args:
            window_size: Size of time window (e.g., '1Y' for 1 year, '6M' for 6 months)
            
        Returns:
            List of (start_date, end_date) tuples
        """
        min_date = self.voting_data['date'].min()
        max_date = self.voting_data['date'].max()
        
        # Create time windows
        windows = []
        current_date = min_date
        
        while current_date < max_date:
            next_date = current_date + pd.Timedelta(window_size)
            windows.append((current_date, min(next_date, max_date)))
            current_date = next_date
        
        return windows
    
    def analyze_alliance_stability(
        self,
        window_size: str = '1Y',
        clustering_method: str = 'hierarchical'
    ) -> Dict:
        """
        Analyze stability of country alliances over time.
        
        Args:
            window_size: Size of time window
            clustering_method: Method for clustering countries
            
        Returns:
            Dictionary with stability metrics
        """
        from ..clustering.country_clustering import CountryClusterer
        
        windows = self.create_time_windows(window_size)
        window_clusters = []
        
        for start_date, end_date in windows:
            # Filter data for this window
            window_data = self.voting_data[
                (self.voting_data['date'] >= start_date) &
                (self.voting_data['date'] < end_date)
            ]
            
            if len(window_data) < 10:  # Skip if too few votes
                continue
            
            # Create vote matrix for this window
            pivot = window_data.pivot_table(
                index='country',
                columns='issue_id',
                values='vote',
                aggfunc='first'
            ).fillna(0)
            
            if len(pivot) < 2:  # Need at least 2 countries
                continue
            
            vote_matrix = pivot.values
            countries = pivot.index.tolist()
            
            # Cluster countries
            clusterer = CountryClusterer()
            labels, _ = clusterer.cluster_countries(
                vote_matrix,
                countries,
                method=clustering_method
            )
            
            window_clusters.append({
                'start_date': start_date,
                'end_date': end_date,
                'countries': countries,
                'labels': labels
            })
        
        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(window_clusters)
        
        return stability_metrics
    
    def _calculate_stability_metrics(
        self,
        window_clusters: List[Dict]
    ) -> Dict:
        """
        Calculate stability metrics from window clusters.
        
        Args:
            window_clusters: List of clustering results per window
            
        Returns:
            Dictionary with stability metrics
        """
        if len(window_clusters) < 2:
            return {'stability_score': 0, 'n_windows': len(window_clusters)}
        
        # Calculate cluster consistency between consecutive windows
        consistency_scores = []
        
        for i in range(len(window_clusters) - 1):
            window1 = window_clusters[i]
            window2 = window_clusters[i + 1]
            
            # Find common countries
            common_countries = set(window1['countries']) & set(window2['countries'])
            
            if len(common_countries) < 2:
                continue
            
            # Get cluster assignments for common countries
            labels1 = []
            labels2 = []
            
            for country in common_countries:
                idx1 = window1['countries'].index(country)
                idx2 = window2['countries'].index(country)
                labels1.append(window1['labels'][idx1])
                labels2.append(window2['labels'][idx2])
            
            # Calculate adjusted Rand index or simple agreement
            agreements = 0
            total_pairs = 0
            
            for j in range(len(labels1)):
                for k in range(j + 1, len(labels1)):
                    total_pairs += 1
                    # Check if countries are in same cluster in both windows
                    same_cluster_1 = labels1[j] == labels1[k]
                    same_cluster_2 = labels2[j] == labels2[k]
                    if same_cluster_1 == same_cluster_2:
                        agreements += 1
            
            if total_pairs > 0:
                consistency = agreements / total_pairs
                consistency_scores.append(consistency)
        
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
        
        return {
            'stability_score': float(avg_consistency),
            'n_windows': len(window_clusters),
            'consistency_scores': [float(s) for s in consistency_scores]
        }
    
    def track_relationship_evolution(
        self,
        country1: str,
        country2: str,
        window_size: str = '1Y'
    ) -> pd.DataFrame:
        """
        Track the evolution of relationship between two countries over time.
        
        Args:
            country1: First country name
            country2: Second country name
            window_size: Size of time window
            
        Returns:
            DataFrame with relationship metrics over time
        """
        windows = self.create_time_windows(window_size)
        evolution_data = []
        
        for start_date, end_date in windows:
            # Filter data for this window
            window_data = self.voting_data[
                (self.voting_data['date'] >= start_date) &
                (self.voting_data['date'] < end_date)
            ]
            
            # Get votes for both countries
            votes1 = window_data[window_data['country'] == country1]
            votes2 = window_data[window_data['country'] == country2]
            
            # Find common issues
            common_issues = set(votes1['issue_id']) & set(votes2['issue_id'])
            
            if len(common_issues) < 5:  # Need minimum votes
                continue
            
            # Calculate agreement
            agreements = 0
            disagreements = 0
            
            for issue_id in common_issues:
                vote1 = votes1[votes1['issue_id'] == issue_id]['vote'].iloc[0]
                vote2 = votes2[votes2['issue_id'] == issue_id]['vote'].iloc[0]
                
                if vote1 == vote2:
                    agreements += 1
                elif (vote1 == 1 and vote2 == -1) or (vote1 == -1 and vote2 == 1):
                    disagreements += 1
            
            total = agreements + disagreements
            if total > 0:
                agreement_rate = agreements / total
                conflict_rate = disagreements / total
            else:
                agreement_rate = 0
                conflict_rate = 0
            
            evolution_data.append({
                'start_date': start_date,
                'end_date': end_date,
                'agreement_rate': agreement_rate,
                'conflict_rate': conflict_rate,
                'n_common_votes': len(common_issues)
            })
        
        return pd.DataFrame(evolution_data)
    
    def identify_trending_issues(
        self,
        window_size: str = '1Y',
        top_k: int = 5
    ) -> Dict[str, List[str]]:
        """
        Identify trending issue clusters over time.
        
        Args:
            window_size: Size of time window
            top_k: Number of top issues per window
            
        Returns:
            Dictionary mapping time windows to trending issues
        """
        windows = self.create_time_windows(window_size)
        trending_issues = {}
        
        for start_date, end_date in windows:
            # Filter data for this window
            window_data = self.voting_data[
                (self.voting_data['date'] >= start_date) &
                (self.voting_data['date'] < end_date)
            ]
            
            if len(window_data) < 10:
                continue
            
            # Count issue frequency
            issue_counts = window_data['issue_id'].value_counts()
            
            # Get top issues
            top_issues = issue_counts.head(top_k).index.tolist()
            
            window_key = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            trending_issues[window_key] = top_issues
        
        return trending_issues
    
    def calculate_temporal_influence(
        self,
        window_size: str = '1Y'
    ) -> pd.DataFrame:
        """
        Calculate country influence over time.
        
        Args:
            window_size: Size of time window
            
        Returns:
            DataFrame with country influence scores over time
        """
        from ..analysis.voting_patterns import VotingAnalyzer
        
        windows = self.create_time_windows(window_size)
        influence_data = []
        
        for start_date, end_date in windows:
            # Filter data for this window
            window_data = self.voting_data[
                (self.voting_data['date'] >= start_date) &
                (self.voting_data['date'] < end_date)
            ]
            
            if len(window_data) < 10:
                continue
            
            # Create vote matrix
            pivot = window_data.pivot_table(
                index='country',
                columns='issue_id',
                values='vote',
                aggfunc='first'
            ).fillna(0)
            
            if len(pivot) < 2:
                continue
            
            vote_matrix = pivot.values
            
            # Calculate influence
            analyzer = VotingAnalyzer(window_data, vote_matrix)
            influence_scores = analyzer.calculate_country_influence()
            
            for country, score in influence_scores.items():
                influence_data.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'country': country,
                    'influence_score': score
                })
        
        return pd.DataFrame(influence_data)
