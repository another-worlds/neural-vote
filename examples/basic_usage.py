"""
Example usage of the Neural Vote library.

This script demonstrates how to use the UN voting analysis pipeline
with sample data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from neural_vote import VotingAnalysisPipeline


def create_sample_data(n_countries=20, n_issues=50, n_time_periods=3):
    """
    Create sample voting data for demonstration.
    
    Args:
        n_countries: Number of countries
        n_issues: Number of issues
        n_time_periods: Number of time periods
        
    Returns:
        DataFrame with sample voting data
    """
    np.random.seed(42)
    
    # Generate country names
    countries = [f"Country_{i:02d}" for i in range(n_countries)]
    
    # Generate issue descriptions
    issue_topics = [
        "Human Rights", "Climate Change", "Nuclear Disarmament",
        "Economic Development", "Refugee Crisis", "Health Policy",
        "Education", "Gender Equality", "Trade Relations", "Conflict Resolution"
    ]
    
    issues = []
    issue_descriptions = []
    for i in range(n_issues):
        topic = issue_topics[i % len(issue_topics)]
        issues.append(f"Issue_{i:03d}")
        issue_descriptions.append(f"{topic} - Resolution {i}")
    
    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = []
    
    data = []
    for period in range(n_time_periods):
        period_start = start_date + timedelta(days=365 * period)
        
        for issue_id, issue_desc in zip(issues, issue_descriptions):
            # Create voting blocs (some countries vote similarly)
            bloc1 = countries[:n_countries//3]
            bloc2 = countries[n_countries//3:2*n_countries//3]
            bloc3 = countries[2*n_countries//3:]
            
            # Assign votes based on issue type and bloc
            issue_type = hash(issue_id) % 3
            
            for country in countries:
                if country in bloc1:
                    if issue_type == 0:
                        vote = 1  # Yes
                    elif issue_type == 1:
                        vote = -1  # No
                    else:
                        vote = 0  # Abstain
                elif country in bloc2:
                    if issue_type == 0:
                        vote = -1
                    elif issue_type == 1:
                        vote = 1
                    else:
                        vote = 1
                else:  # bloc3
                    if issue_type == 0:
                        vote = 0
                    elif issue_type == 1:
                        vote = 0
                    else:
                        vote = -1
                
                # Add some randomness
                if np.random.random() < 0.2:
                    vote = np.random.choice([-1, 0, 1])
                
                vote_date = period_start + timedelta(days=np.random.randint(0, 300))
                
                data.append({
                    'country': country,
                    'issue_id': issue_id,
                    'issue_description': issue_desc,
                    'vote': vote,
                    'date': vote_date
                })
    
    return pd.DataFrame(data)


def main():
    """Main example demonstrating the voting analysis pipeline."""
    
    print("=" * 70)
    print("UN VOTING RELATIONSHIPS ANALYSIS - EXAMPLE")
    print("=" * 70)
    print()
    
    # Create sample data
    print("Creating sample voting data...")
    sample_data = create_sample_data(n_countries=20, n_issues=50, n_time_periods=3)
    print(f"✓ Created {len(sample_data)} voting records")
    print(f"  - Countries: {sample_data['country'].nunique()}")
    print(f"  - Issues: {sample_data['issue_id'].nunique()}")
    print(f"  - Date range: {sample_data['date'].min()} to {sample_data['date'].max()}")
    print()
    
    # Initialize pipeline
    pipeline = VotingAnalysisPipeline()
    
    # Run full analysis
    results = pipeline.run_full_pipeline(
        dataframe=sample_data,
        n_issue_clusters=5,
        n_country_clusters=3,
        temporal_window='1Y'
    )
    
    # Display summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    summary = pipeline.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Display detailed results
    print("\n" + "=" * 70)
    print("ISSUE CLUSTERS")
    print("=" * 70)
    for cluster_id, info in results['issue_clusters']['info'].items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {info['size']} issues ({info['percentage']:.1f}%)")
        print(f"  Example issues:")
        for issue in info['issues'][:3]:
            print(f"    - {issue}")
    
    print("\n" + "=" * 70)
    print("COUNTRY ALLIANCES")
    print("=" * 70)
    for cluster_id, info in results['country_clusters']['info'].items():
        print(f"\nAlliance {cluster_id}:")
        print(f"  Size: {info['size']} countries ({info['percentage']:.1f}%)")
        print(f"  Members: {', '.join(info['countries'][:5])}")
        if info['size'] > 5:
            print(f"    ... and {info['size'] - 5} more")
    
    print("\n" + "=" * 70)
    print("TOP CONTROVERSIAL ISSUES")
    print("=" * 70)
    for i, (issue_id, score) in enumerate(results['voting_analysis']['controversial_issues'][:5], 1):
        print(f"{i}. {issue_id} (controversy score: {score:.3f})")
    
    print("\n" + "=" * 70)
    print("COUNTRY INFLUENCE SCORES (Top 5)")
    print("=" * 70)
    influence = results['voting_analysis']['country_influence']
    top_influential = sorted(influence.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (country, score) in enumerate(top_influential, 1):
        print(f"{i}. {country}: {score:.3f}")
    
    if results['temporal_analysis']:
        print("\n" + "=" * 70)
        print("TEMPORAL DYNAMICS")
        print("=" * 70)
        print(f"Alliance stability score: {results['temporal_analysis']['stability_metrics']['stability_score']:.3f}")
        print(f"Number of time windows analyzed: {results['temporal_analysis']['stability_metrics']['n_windows']}")
    
    print("\n" + "=" * 70)
    print("Analysis complete! Results saved in 'results' dictionary.")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
