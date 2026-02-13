# Neural Vote

United Nations voting relationships analysis neural model.

## Overview

Neural Vote is a comprehensive framework for analyzing UN voting patterns and relationships between countries. It uses machine learning and data analysis techniques to:

1. **Load and preprocess** UN voting data
2. **Semantically cluster** issues using neural embeddings
3. **Analyze voting interrelations** with importance and conflict metrics
4. **Cluster countries** into voting alliances
5. **Track temporal dynamics** of voting relationships over time

## Features

- **Data Loading**: Flexible data loading from CSV files or pandas DataFrames
- **Issue Clustering**: Semantic clustering using sentence transformers or TF-IDF
- **Country Clustering**: Alliance detection based on voting patterns
- **Voting Analysis**: 
  - Issue importance scoring
  - Conflict/agreement matrices
  - Controversial issue identification
  - Country influence metrics
- **Temporal Analysis**:
  - Alliance stability over time
  - Relationship evolution tracking
  - Trending issue identification
  - Temporal influence metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/another-worlds/neural-vote.git
cd neural-vote

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
from neural_vote import VotingAnalysisPipeline
import pandas as pd

# Load your voting data
# Expected columns: country, issue_id, vote, date (optional), issue_description (optional)
data = pd.read_csv('your_voting_data.csv')

# Initialize pipeline
pipeline = VotingAnalysisPipeline()

# Run complete analysis
results = pipeline.run_full_pipeline(
    dataframe=data,
    n_issue_clusters=5,      # Number of issue clusters (auto if None)
    n_country_clusters=3,    # Number of country alliances (auto if None)
    temporal_window='1Y'     # Time window for temporal analysis
)

# Get summary
summary = pipeline.get_summary()
print(summary)
```

## Data Format

Your voting data should include the following columns:

- **country** (required): Country name
- **issue_id** (required): Unique identifier for the issue
- **vote** (required): Vote value (1=yes, 0=abstain, -1=no)
- **date** (optional): Date of the vote (required for temporal analysis)
- **issue_description** (optional): Description of the issue (used for semantic clustering)

Example:

```csv
country,issue_id,issue_description,vote,date
USA,I001,Human Rights Resolution,1,2020-01-15
China,I001,Human Rights Resolution,-1,2020-01-15
UK,I001,Human Rights Resolution,1,2020-01-15
```

## Usage Examples

### Basic Analysis

```python
from neural_vote import VotingAnalysisPipeline

pipeline = VotingAnalysisPipeline()

# Load data
pipeline.load_data(filepath='voting_data.csv')

# Step-by-step analysis
issue_clusters = pipeline.cluster_issues(n_clusters=5)
voting_analysis = pipeline.analyze_voting_patterns()
country_alliances = pipeline.cluster_countries(n_clusters=3)
temporal_dynamics = pipeline.analyze_temporal_dynamics(window_size='1Y')
```

### Advanced Usage

```python
from neural_vote import (
    VotingDataLoader,
    IssueClusterer,
    CountryClusterer,
    VotingAnalyzer,
    TemporalAnalyzer
)

# Load data
loader = VotingDataLoader()
data = loader.load_from_csv('voting_data.csv')
vote_matrix = loader.create_vote_matrix()

# Cluster issues
issue_clusterer = IssueClusterer()
issue_descriptions = data['issue_description'].unique().tolist()
labels, cluster_info = issue_clusterer.cluster_issues(issue_descriptions)

# Cluster countries
country_clusterer = CountryClusterer()
labels, alliances = country_clusterer.cluster_countries(
    vote_matrix,
    loader.countries,
    method='hierarchical'
)

# Analyze voting patterns
analyzer = VotingAnalyzer(data, vote_matrix)
importance_scores = analyzer.calculate_issue_importance()
conflict_matrix = analyzer.calculate_conflict_matrix()
controversial_issues = analyzer.get_most_controversial_issues()

# Temporal analysis
temporal = TemporalAnalyzer(data)
stability = temporal.analyze_alliance_stability(window_size='1Y')
relationship = temporal.track_relationship_evolution('USA', 'UK', '1Y')
```

## Running the Example

An example script with sample data is provided:

```bash
cd examples
python basic_usage.py
```

This will:
- Generate sample voting data
- Run the complete analysis pipeline
- Display results including:
  - Issue clusters
  - Country alliances
  - Controversial issues
  - Country influence scores
  - Temporal dynamics

## Architecture

```
neural_vote/
├── data/
│   └── loader.py          # Data loading and preprocessing
├── clustering/
│   ├── issue_clustering.py      # Semantic issue clustering
│   └── country_clustering.py    # Country alliance detection
├── analysis/
│   ├── voting_patterns.py       # Voting pattern analysis
│   └── temporal_dynamics.py     # Temporal relationship tracking
└── pipeline.py            # Main orchestration pipeline
```

## API Reference

### VotingAnalysisPipeline

Main pipeline class that orchestrates the complete analysis.

**Methods:**
- `load_data(filepath, dataframe)`: Load voting data
- `cluster_issues(n_clusters, method)`: Cluster issues semantically
- `analyze_voting_patterns()`: Analyze voting interrelations
- `cluster_countries(n_clusters, method)`: Cluster countries into alliances
- `analyze_temporal_dynamics(window_size)`: Analyze temporal dynamics
- `run_full_pipeline(...)`: Run complete analysis
- `get_summary()`: Get analysis summary

### VotingDataLoader

Data loading and preprocessing utilities.

**Methods:**
- `load_from_csv(filepath)`: Load from CSV file
- `load_from_dataframe(df)`: Load from DataFrame
- `create_vote_matrix()`: Create country x issue voting matrix
- `get_country_votes(country)`: Get votes for a country
- `get_issue_votes(issue_id)`: Get votes for an issue
- `filter_by_date_range(start, end)`: Filter by date range

### IssueClusterer

Semantic clustering of issues.

**Methods:**
- `encode_issues(descriptions)`: Encode issues to embeddings
- `cluster_issues(descriptions, n_clusters, method)`: Cluster issues
- `get_cluster_label(issue_index)`: Get cluster for an issue
- `assign_votes_to_clusters(voting_data, issue_ids, descriptions)`: Assign clusters

### CountryClusterer

Country alliance detection.

**Methods:**
- `compute_similarity_matrix(vote_matrix, metric)`: Compute country similarities
- `cluster_countries(vote_matrix, countries, n_clusters, method)`: Cluster countries
- `get_country_alliance(country)`: Get alliance for a country
- `get_similar_countries(country, top_k)`: Get most similar countries

### VotingAnalyzer

Voting pattern analysis.

**Methods:**
- `calculate_issue_importance()`: Calculate importance scores
- `calculate_conflict_matrix()`: Calculate conflict between countries
- `calculate_agreement_matrix()`: Calculate agreement between countries
- `analyze_issue_correlations()`: Analyze issue correlations
- `get_most_controversial_issues(top_k)`: Get controversial issues
- `calculate_country_influence()`: Calculate country influence scores

### TemporalAnalyzer

Temporal dynamics analysis.

**Methods:**
- `create_time_windows(window_size)`: Create time windows
- `analyze_alliance_stability(window_size)`: Analyze alliance stability
- `track_relationship_evolution(country1, country2, window_size)`: Track relationships
- `identify_trending_issues(window_size, top_k)`: Identify trending issues
- `calculate_temporal_influence(window_size)`: Calculate influence over time

## Requirements

- Python >= 3.8
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- torch >= 2.0.0
- transformers >= 4.30.0
- sentence-transformers >= 2.2.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scipy >= 1.7.0
- networkx >= 2.6.0
- tqdm >= 4.62.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Citation

If you use this software in your research, please cite:

```bibtex
@software{neural_vote,
  title = {Neural Vote: UN Voting Relationships Analysis},
  author = {Another Worlds},
  year = {2024},
  url = {https://github.com/another-worlds/neural-vote}
}
```

## Contact

For questions or feedback, please open an issue on GitHub.