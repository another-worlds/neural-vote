"""
Simple test to verify the basic functionality of the neural_vote package.
"""
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_simple_test_data():
    """Create a small test dataset."""
    np.random.seed(42)
    
    countries = ['USA', 'UK', 'France', 'China', 'Russia', 'Brazil']
    issues = ['HR001', 'CC002', 'ND003', 'ED004', 'RC005']
    issue_descs = [
        'Human Rights Resolution 1',
        'Climate Change Agreement',
        'Nuclear Disarmament Treaty',
        'Economic Development Plan',
        'Refugee Crisis Response'
    ]
    
    data = []
    start_date = datetime(2020, 1, 1)
    
    for i, (issue_id, issue_desc) in enumerate(zip(issues, issue_descs)):
        for j, country in enumerate(countries):
            # Create voting patterns: some countries align more than others
            if country in ['USA', 'UK', 'France']:
                vote = 1 if i % 2 == 0 else -1
            elif country in ['China', 'Russia']:
                vote = -1 if i % 2 == 0 else 1
            else:
                vote = 0
            
            # Add some randomness
            if np.random.random() < 0.2:
                vote = np.random.choice([-1, 0, 1])
            
            data.append({
                'country': country,
                'issue_id': issue_id,
                'issue_description': issue_desc,
                'vote': vote,
                'date': start_date + timedelta(days=i*30 + j*5)
            })
    
    return pd.DataFrame(data)


def test_basic_functionality():
    """Test basic functionality without heavy dependencies."""
    print("=" * 60)
    print("NEURAL VOTE - BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    print()
    
    # Import modules
    print("1. Testing imports...")
    try:
        from neural_vote import (
            VotingDataLoader,
            VotingAnalysisPipeline
        )
        print("   ✓ Core imports successful")
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        return False
    
    # Create test data
    print("\n2. Creating test data...")
    data = create_simple_test_data()
    print(f"   ✓ Created {len(data)} voting records")
    print(f"   - Countries: {data['country'].nunique()}")
    print(f"   - Issues: {data['issue_id'].nunique()}")
    
    # Test data loader
    print("\n3. Testing data loader...")
    try:
        loader = VotingDataLoader()
        voting_data = loader.load_from_dataframe(data)
        vote_matrix = loader.create_vote_matrix()
        print(f"   ✓ Data loaded successfully")
        print(f"   - Vote matrix shape: {vote_matrix.shape}")
        print(f"   - Countries: {len(loader.countries)}")
        print(f"   - Issues: {len(loader.issues)}")
    except Exception as e:
        print(f"   ✗ Data loader failed: {e}")
        return False
    
    # Test pipeline initialization
    print("\n4. Testing pipeline...")
    try:
        pipeline = VotingAnalysisPipeline()
        pipeline.load_data(dataframe=data)
        print(f"   ✓ Pipeline initialized successfully")
    except Exception as e:
        print(f"   ✗ Pipeline failed: {e}")
        return False
    
    # Test country clustering (without neural embeddings)
    print("\n5. Testing country clustering...")
    try:
        from neural_vote import CountryClusterer
        clusterer = CountryClusterer()
        labels, info = clusterer.cluster_countries(
            vote_matrix,
            loader.countries,
            n_clusters=2,
            method='kmeans'
        )
        print(f"   ✓ Country clustering successful")
        print(f"   - Number of clusters: {len(info)}")
        for cluster_id, cluster_info in info.items():
            print(f"   - Cluster {cluster_id}: {cluster_info['countries']}")
    except Exception as e:
        print(f"   ✗ Country clustering failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test voting analysis
    print("\n6. Testing voting analysis...")
    try:
        from neural_vote import VotingAnalyzer
        analyzer = VotingAnalyzer(voting_data, vote_matrix)
        
        importance = analyzer.calculate_issue_importance()
        conflict_matrix = analyzer.calculate_conflict_matrix()
        agreement_matrix = analyzer.calculate_agreement_matrix()
        controversial = analyzer.get_most_controversial_issues(top_k=3)
        influence = analyzer.calculate_country_influence()
        
        print(f"   ✓ Voting analysis successful")
        print(f"   - Issue importance calculated: {len(importance)} issues")
        print(f"   - Conflict matrix shape: {conflict_matrix.shape}")
        print(f"   - Top controversial issue: {controversial[0][0]}")
        print(f"   - Most influential country: {max(influence.items(), key=lambda x: x[1])[0]}")
    except Exception as e:
        print(f"   ✗ Voting analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test temporal analysis (if date column exists)
    print("\n7. Testing temporal analysis...")
    try:
        from neural_vote import TemporalAnalyzer
        temporal = TemporalAnalyzer(voting_data)
        
        windows = temporal.create_time_windows(window_size='6M')
        print(f"   ✓ Temporal analysis successful")
        print(f"   - Time windows created: {len(windows)}")
        
        # Test relationship tracking
        evolution = temporal.track_relationship_evolution('USA', 'UK', '6M')
        print(f"   - Relationship evolution tracked: {len(evolution)} periods")
    except Exception as e:
        print(f"   ✗ Temporal analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
