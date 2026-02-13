"""
Main pipeline orchestrator for UN voting analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

from .data.loader import VotingDataLoader
from .clustering.issue_clustering import IssueClusterer
from .clustering.country_clustering import CountryClusterer
from .analysis.voting_patterns import VotingAnalyzer
from .analysis.temporal_dynamics import TemporalAnalyzer


class VotingAnalysisPipeline:
    """
    Main pipeline for UN voting relationships analysis.
    
    This pipeline orchestrates all steps:
    1. Load voting data
    2. Cluster issues semantically
    3. Analyze voting interrelations
    4. Cluster countries into alliances
    5. Analyze temporal dynamics
    """
    
    def __init__(self):
        self.data_loader = VotingDataLoader()
        self.issue_clusterer = IssueClusterer()
        self.country_clusterer = CountryClusterer()
        self.voting_data = None
        self.vote_matrix = None
        self.results = {}
    
    def load_data(self, filepath: str = None, dataframe: pd.DataFrame = None):
        """
        Load voting data from file or DataFrame.
        
        Args:
            filepath: Path to CSV file (optional)
            dataframe: DataFrame with voting data (optional)
        """
        if filepath:
            self.voting_data = self.data_loader.load_from_csv(filepath)
        elif dataframe is not None:
            self.voting_data = self.data_loader.load_from_dataframe(dataframe)
        else:
            raise ValueError("Either filepath or dataframe must be provided.")
        
        self.vote_matrix = self.data_loader.create_vote_matrix()
        print(f"✓ Loaded data: {len(self.data_loader.countries)} countries, "
              f"{len(self.data_loader.issues)} issues")
    
    def cluster_issues(
        self,
        n_clusters: Optional[int] = None,
        method: str = 'kmeans'
    ) -> Dict:
        """
        Cluster issues semantically (Step 2).
        
        Args:
            n_clusters: Number of clusters (if None, auto-determine)
            method: Clustering method
            
        Returns:
            Dictionary with clustering results
        """
        print("Step 2: Clustering issues semantically...")
        
        # Get issue descriptions
        issue_descriptions = []
        for issue_id in self.data_loader.issues:
            # Get description if available, otherwise use issue_id
            issue_data = self.voting_data[self.voting_data['issue_id'] == issue_id]
            if 'issue_description' in self.voting_data.columns:
                desc = issue_data['issue_description'].iloc[0]
            else:
                desc = str(issue_id)
            issue_descriptions.append(desc)
        
        # Cluster issues
        labels, cluster_info = self.issue_clusterer.cluster_issues(
            issue_descriptions,
            n_clusters=n_clusters,
            method=method
        )
        
        # Assign clusters to voting data
        self.voting_data = self.issue_clusterer.assign_votes_to_clusters(
            self.voting_data,
            self.data_loader.issues,
            issue_descriptions
        )
        
        self.results['issue_clusters'] = {
            'labels': labels,
            'info': cluster_info,
            'n_clusters': len(cluster_info)
        }
        
        print(f"✓ Created {len(cluster_info)} issue clusters")
        return self.results['issue_clusters']
    
    def analyze_voting_patterns(self) -> Dict:
        """
        Analyze voting interrelations (Step 3).
        
        Returns:
            Dictionary with analysis results
        """
        print("Step 3: Analyzing voting interrelations...")
        
        analyzer = VotingAnalyzer(self.voting_data, self.vote_matrix)
        
        # Calculate various metrics
        importance_scores = analyzer.calculate_issue_importance()
        conflict_matrix = analyzer.calculate_conflict_matrix()
        agreement_matrix = analyzer.calculate_agreement_matrix()
        controversial_issues = analyzer.get_most_controversial_issues(top_k=10)
        country_influence = analyzer.calculate_country_influence()
        
        # Analyze cluster voting patterns if clusters exist
        cluster_patterns = None
        if 'cluster' in self.voting_data.columns:
            cluster_labels = self.voting_data.groupby('issue_id')['cluster'].first().values
            cluster_patterns = analyzer.analyze_cluster_voting_patterns(
                cluster_labels,
                cluster_type='issue'
            )
        
        self.results['voting_analysis'] = {
            'importance_scores': importance_scores,
            'conflict_matrix': conflict_matrix,
            'agreement_matrix': agreement_matrix,
            'controversial_issues': controversial_issues,
            'country_influence': country_influence,
            'cluster_patterns': cluster_patterns
        }
        
        print(f"✓ Analyzed voting patterns")
        print(f"  - Top controversial issue: {controversial_issues[0][0]}")
        print(f"  - Most influential country: {max(country_influence.items(), key=lambda x: x[1])[0]}")
        
        return self.results['voting_analysis']
    
    def cluster_countries(
        self,
        n_clusters: Optional[int] = None,
        method: str = 'hierarchical'
    ) -> Dict:
        """
        Cluster countries into alliances (Step 4).
        
        Args:
            n_clusters: Number of clusters (if None, auto-determine)
            method: Clustering method
            
        Returns:
            Dictionary with clustering results
        """
        print("Step 4: Clustering countries into alliances...")
        
        labels, cluster_info = self.country_clusterer.cluster_countries(
            self.vote_matrix,
            self.data_loader.countries,
            n_clusters=n_clusters,
            method=method
        )
        
        self.results['country_clusters'] = {
            'labels': labels,
            'info': cluster_info,
            'n_clusters': len(cluster_info),
            'similarity_matrix': self.country_clusterer.similarity_matrix
        }
        
        print(f"✓ Created {len(cluster_info)} country alliances")
        for cluster_id, info in cluster_info.items():
            print(f"  - Alliance {cluster_id}: {info['size']} countries")
        
        return self.results['country_clusters']
    
    def analyze_temporal_dynamics(
        self,
        window_size: str = '1Y'
    ) -> Dict:
        """
        Analyze temporal dynamics (Step 5).
        
        Args:
            window_size: Size of time window for analysis
            
        Returns:
            Dictionary with temporal analysis results
        """
        print("Step 5: Analyzing temporal dynamics...")
        
        # Check if date column exists
        if 'date' not in self.voting_data.columns:
            warnings.warn("No date column in voting data. Skipping temporal analysis.")
            self.results['temporal_analysis'] = None
            return None
        
        temporal_analyzer = TemporalAnalyzer(self.voting_data)
        
        # Analyze alliance stability
        stability_metrics = temporal_analyzer.analyze_alliance_stability(
            window_size=window_size
        )
        
        # Identify trending issues
        trending_issues = temporal_analyzer.identify_trending_issues(
            window_size=window_size
        )
        
        # Calculate temporal influence
        temporal_influence = temporal_analyzer.calculate_temporal_influence(
            window_size=window_size
        )
        
        self.results['temporal_analysis'] = {
            'stability_metrics': stability_metrics,
            'trending_issues': trending_issues,
            'temporal_influence': temporal_influence
        }
        
        print(f"✓ Analyzed temporal dynamics")
        print(f"  - Alliance stability score: {stability_metrics.get('stability_score', 0):.3f}")
        print(f"  - Number of time windows: {stability_metrics.get('n_windows', 0)}")
        
        return self.results['temporal_analysis']
    
    def run_full_pipeline(
        self,
        filepath: str = None,
        dataframe: pd.DataFrame = None,
        n_issue_clusters: Optional[int] = None,
        n_country_clusters: Optional[int] = None,
        temporal_window: str = '1Y'
    ) -> Dict:
        """
        Run the complete analysis pipeline.
        
        Args:
            filepath: Path to data file
            dataframe: DataFrame with data
            n_issue_clusters: Number of issue clusters
            n_country_clusters: Number of country clusters
            temporal_window: Size of temporal analysis window
            
        Returns:
            Dictionary with all results
        """
        print("=" * 60)
        print("UN VOTING RELATIONSHIPS ANALYSIS")
        print("=" * 60)
        
        # Step 1: Load data
        print("\nStep 1: Loading voting data...")
        self.load_data(filepath=filepath, dataframe=dataframe)
        
        # Step 2: Cluster issues
        self.cluster_issues(n_clusters=n_issue_clusters)
        
        # Step 3: Analyze voting patterns
        self.analyze_voting_patterns()
        
        # Step 4: Cluster countries
        self.cluster_countries(n_clusters=n_country_clusters)
        
        # Step 5: Analyze temporal dynamics
        self.analyze_temporal_dynamics(window_size=temporal_window)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        
        return self.results
    
    def get_summary(self) -> Dict:
        """
        Get a summary of the analysis results.
        
        Returns:
            Dictionary with summary information
        """
        if not self.results:
            return {"error": "No analysis has been run yet."}
        
        summary = {
            'n_countries': len(self.data_loader.countries),
            'n_issues': len(self.data_loader.issues),
            'n_votes': len(self.voting_data)
        }
        
        if 'issue_clusters' in self.results:
            summary['n_issue_clusters'] = self.results['issue_clusters']['n_clusters']
        
        if 'country_clusters' in self.results:
            summary['n_country_clusters'] = self.results['country_clusters']['n_clusters']
        
        if 'voting_analysis' in self.results:
            va = self.results['voting_analysis']
            if va['controversial_issues']:
                summary['most_controversial_issue'] = va['controversial_issues'][0][0]
        
        if 'temporal_analysis' in self.results and self.results['temporal_analysis']:
            ta = self.results['temporal_analysis']
            summary['alliance_stability'] = ta['stability_metrics'].get('stability_score', 0)
        
        return summary
