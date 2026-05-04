"""
Basic example of weight retrieval for graph traversal.

This example demonstrates how to use the WeightRetriever to compute
relationship weights for a query metric using Bayesian updates.
"""

import numpy as np
import pandas as pd
from shared.retrieval_package.core import WeightRetriever
from shared.retrieval_package.config import RetrievalConfig
from shared.retrieval_package.utils import get_prior_matrix, compute_correlation_stats


def main():
    """Run basic weight retrieval example."""
    print("=" * 60)
    print("Weight Retrieval Example")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load LLM Prior Matrix
    # =========================================================================
    print("\n[Step 1] Loading LLM prior matrix from knowledge graph...")

    # In practice, load from your graph construction output
    # llm_prior_df = get_prior_matrix("experiment_1")

    # For this example, create synthetic prior
    metrics = ['Heart Rate', 'Sleep Duration', 'Step Count', 'Resting HR', 'HRV']
    llm_prior_df = pd.DataFrame(
        data=np.random.rand(len(metrics), len(metrics)),
        index=metrics,
        columns=metrics
    )
    # Set diagonal to 1 (self-relationship)
    np.fill_diagonal(llm_prior_df.values, 1.0)

    print(f"Loaded prior matrix with {len(metrics)} metrics")

    # =========================================================================
    # Step 2: Compute Population Correlations
    # =========================================================================
    print("\n[Step 2] Computing population correlations...")

    # Create synthetic population data
    np.random.seed(42)
    n_samples_pop = 100
    pop_data = pd.DataFrame(
        {metric: np.random.randn(n_samples_pop) for metric in metrics}
    )

    # Compute correlations
    rel_pop, p_pop, sample_size_pop = compute_correlation_stats(
        pop_data,
        metrics,
        corr_type='spearman'
    )

    print(f"Computed population correlations from {n_samples_pop} samples")

    # =========================================================================
    # Step 3: Compute Individual Correlations
    # =========================================================================
    print("\n[Step 3] Computing individual correlations...")

    # Create synthetic individual data with dates
    n_samples_ind = 50
    dates = pd.date_range(end='2024-01-15', periods=n_samples_ind, freq='D')
    ind_data = pd.DataFrame(
        {metric: np.random.randn(n_samples_ind) for metric in metrics}
    )
    ind_data['date'] = dates

    # Compute individual correlations (excluding date column)
    rel_ind, p_ind, sample_size_ind = compute_correlation_stats(
        ind_data.drop(columns=['date']),
        metrics,
        corr_type='spearman'
    )

    print(f"Computed individual correlations from {n_samples_ind} samples")

    # =========================================================================
    # Step 4: Prepare Relationship Info
    # =========================================================================
    print("\n[Step 4] Preparing relationship information...")

    rel_info = {
        'rel_pop_all': rel_pop,
        'rel_var_pop_all': None,  # Computed from sample size
        'rel_pop_sample_size_all': sample_size_pop,
        'rel_ind_all': rel_ind,
        'rel_var_ind_all': None,  # Computed from sample size
        'rel_ind_sample_size_all': sample_size_ind,
        'numeric_metrics': metrics,
        'data_associated_metrics': metrics,  # All metrics have data
    }

    # =========================================================================
    # Step 5: Configure Hyperparameters
    # =========================================================================
    print("\n[Step 5] Configuring hyperparameters...")

    hyperparams = RetrievalConfig.get_hyperparameters(
        rel_type='spearman',
        min_samples=4,
        full_rel_only=False,
        beta=0.5,  # Equal weight to global and local signals
        t_global=1.0,
        t_local=0.7,
    )

    print(f"Using hyperparameters: {hyperparams}")

    # =========================================================================
    # Step 6: Define Query Context
    # =========================================================================
    print("\n[Step 6] Defining query context...")

    query_info = {
        'query_date': '2024-01-15',
        'time_granularity': 7,  # Last 7 days
        'openness': 0.5,  # Medium openness
    }

    query_metric = 'Heart Rate'
    print(f"Query metric: {query_metric}")
    print(f"Query date: {query_info['query_date']}")
    print(f"Time window: {query_info['time_granularity']} days")
    print(f"Openness: {query_info['openness']}")

    # =========================================================================
    # Step 7: Run Weight Retrieval
    # =========================================================================
    print("\n[Step 7] Running weight retrieval...")

    retriever = WeightRetriever(
        metric_to_check=query_metric,
        rel_info=rel_info,
        llm_prior_df=llm_prior_df,
        hyperparameter=hyperparams,
        ind_df=ind_data,
        query_info=query_info,
    )

    # Execute weight computation
    weights_df, abnormality_df = retriever.run()

    print("Weight computation completed!")

    # =========================================================================
    # Step 8: Analyze Results
    # =========================================================================
    print("\n[Step 8] Analyzing results...")
    print("\n" + "=" * 60)
    print("TOP RELATED METRICS")
    print("=" * 60)

    # Sort by final weight
    top_related = weights_df.sort_values(by='weight_final', ascending=False)

    print("\nMetric                 Weight_Final  LLM_Prior  Pop_Corr  Ind_Corr")
    print("-" * 60)
    for idx, row in top_related.head(10).iterrows():
        print(f"{idx:20s}  {row['weight_final']:8.4f}  "
              f"{row['llm_prior']:8.4f}  "
              f"{row.get('rel_pop', np.nan):8.4f}  "
              f"{row.get('rel_ind', np.nan):8.4f}")

    # Show abnormality scores
    if not abnormality_df['abnormality'].isna().all():
        print("\n" + "=" * 60)
        print("ABNORMALITY SCORES (Recent 7 Days)")
        print("=" * 60)
        print("\nMetric                 Abnormality")
        print("-" * 60)
        abnormal_sorted = abnormality_df.sort_values(by='abnormality', ascending=False)
        for idx, row in abnormal_sorted.head(10).iterrows():
            if not np.isnan(row['abnormality']):
                print(f"{idx:20s}  {row['abnormality']:8.4f}")

    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Total metrics evaluated: {len(weights_df)}")
    print(f"Average final weight: {weights_df['weight_final'].mean():.4f}")
    print(f"Weight range: [{weights_df['weight_final'].min():.4f}, "
          f"{weights_df['weight_final'].max():.4f}]")

    # Compare weight sources
    if 'weight_local' in weights_df.columns:
        valid_local = weights_df['weight_local'].dropna()
        if len(valid_local) > 0:
            print(f"\nMetrics with local weights: {len(valid_local)}")
            print(f"Average local weight: {valid_local.mean():.4f}")

    print(f"Average global weight: {weights_df['weight_global'].mean():.4f}")

    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
