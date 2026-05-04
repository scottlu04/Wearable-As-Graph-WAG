"""
Weight retrieval and computation for graph traversal.

This module implements the WeightRetriever class which computes relationship
weights between metrics using Bayesian updating of LLM priors with population
and individual correlations, combined with local abnormality signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings

from ..config import RetrievalConfig
from ..utils import (
    fisher_transform,
    update_prior,
    blend_openness_abnormality,
    detect_anomalies,
)


class WeightRetriever:
    """
    Compute relationship weights for graph traversal using Bayesian updates.

    This class implements a sophisticated weight computation system that:
    1. Starts with LLM prior beliefs about metric relationships
    2. Updates with population-level correlation statistics
    3. Further updates with individual-level correlations
    4. Combines with local abnormality signals based on query openness
    5. Produces final weights for graph traversal

    The weights guide which related metrics to retrieve when answering queries.
    """

    def __init__(
        self,
        metric_to_check: str,
        rel_info: Dict,
        llm_prior_df: pd.DataFrame,
        hyperparameter: Dict,
        ind_df: Optional[pd.DataFrame] = None,
        query_info: Optional[Dict] = None,
    ):
        """
        Initialize weight retriever for a specific query metric.

        Args:
            metric_to_check: Name of the metric being queried
            rel_info: Dictionary containing relationship statistics:
                - rel_pop_all: Population correlation matrix
                - rel_var_pop_all: Population correlation variance matrix
                - rel_pop_sample_size_all: Population sample sizes
                - rel_ind_all: Individual correlation matrix
                - rel_var_ind_all: Individual correlation variance matrix
                - rel_ind_sample_size_all: Individual sample sizes
                - numeric_metrics: List of all numeric metric names
                - data_associated_metrics: List of data-associated metric names
            llm_prior_df: DataFrame with LLM prior relationship weights
            hyperparameter: Dictionary of hyperparameters (see RetrievalConfig)
            ind_df: Individual's data DataFrame (optional, for abnormality detection)
            query_info: Query information including:
                - query_date: Date of query
                - time_granularity: Time window (1/7/14/30/60/all days)
                - openness: Query openness score (0.0-1.0)
        """
        # Set random state for reproducibility
        self._rng = np.random.RandomState(42)

        # Store query context
        self.query_metric = metric_to_check
        self.rel_type = hyperparameter['rel_type']
        self.full_rel_only = hyperparameter['full_rel_only']

        # Store relationship information
        self.update_rel_info(rel_info)

        # Store prior and hyperparameters
        self.llm_prior_df = llm_prior_df
        self.hyper = hyperparameter

        # Store individual data and query info
        self.ind_df = ind_df.copy() if ind_df is not None else None
        self.query_info = query_info

        # Initialize results DataFrame
        self.res = self._initialize_result()

        # Track query metric's abnormality
        self.query_abnormality = np.nan

    def _initialize_result(self) -> pd.DataFrame:
        """
        Initialize results DataFrame with LLM priors.

        Returns:
            DataFrame indexed by data-associated metrics with llm_prior column
        """
        res = pd.DataFrame(index=self.data_associated_metrics)

        # Extract LLM priors for query metric
        res['llm_prior'] = self.llm_prior_df.loc[self.data_associated_metrics, self.query_metric]

        # Z-normalize LLM priors
        valid_prior = res['llm_prior'].dropna()
        if not valid_prior.empty:
            mean_prior = valid_prior.mean()
            std_prior = valid_prior.std()

            if std_prior != 0 and not np.isnan(std_prior):
                res['llm_prior'] = (res['llm_prior'] - mean_prior) / std_prior

        # Apply sigmoid transform to map to [0, 1]
        res['llm_prior'] = self._apply_sigmoid(res['llm_prior'])

        return res

    def _apply_sigmoid(self, x: pd.Series) -> pd.Series:
        """
        Apply sigmoid transformation with temperature parameter.

        Args:
            x: Input series

        Returns:
            Sigmoid-transformed series in [0, 1]
        """
        return 1 / (1 + np.exp(-self.hyper['t_global'] * x))

    def _handle_abnormality(self) -> None:
        """
        Detect abnormalities in recent data and compute local weights.

        Uses z-scores to identify metrics with unusual recent values.
        Blends with query openness to determine local weights.
        """
        if self.ind_df is None:
            return

        # Get numeric columns for analysis
        used_cols = self.numeric_metrics + ['date']
        par_df = self.ind_df[used_cols].copy()

        # Detect anomalies in recent time window
        anomaly = detect_anomalies(
            par_df,
            self.query_info['query_date'],
            self.query_info['time_granularity']
        )

        # Map anomaly scores to result DataFrame
        self.res['recent_abnormality'] = self.res.index.map(anomaly)

        # Compute local weight by blending with openness
        if np.isnan(self.res['recent_abnormality']).all():
            self.res['weight_local'] = np.nan
        else:
            _, _, weight = blend_openness_abnormality(
                self.res['recent_abnormality'],
                self.query_info['openness'],
                self.hyper['t_local']
            )
            self.res['weight_local'] = weight

    def _calculate_correlation_features(self) -> None:
        """
        Calculate correlation-based features and perform Bayesian updates.

        Processes population and individual correlations with:
        - Fisher z-transformation
        - Z-normalization
        - Sigmoid transformation
        - Sample size filtering
        """
        if self.rel_type not in ['spearman', 'pearson']:
            return

        # === Population Correlation ===
        if self.rel_pop_all is not None:
            self.res['rel_pop'] = self.rel_pop_all[self.query_metric].abs()
        else:
            self.res['rel_pop'] = np.nan

        # Population sample size and variance
        if self.rel_pop_sample_size_all is not None:
            self.res['pop_sample_size'] = self.rel_pop_sample_size_all[self.query_metric]
            # Fisher z-transformation variance: 1/sqrt(n-3)
            self.res['var_pop'] = 1 / np.sqrt(self.res['pop_sample_size'].clip(lower=4) - 3)
        else:
            self.res['pop_sample_size'] = np.nan
            self.res['var_pop'] = np.nan

        # === Individual Correlation ===
        if self.rel_ind_all is not None:
            self.res['rel_ind'] = self.rel_ind_all[self.query_metric].abs()
        else:
            self.res['rel_ind'] = np.nan

        # Individual sample size and variance
        if self.rel_ind_sample_size_all is not None:
            self.res['ind_sample_size'] = self.rel_ind_sample_size_all[self.query_metric]
            self.res['var_ind'] = 1 / np.sqrt(self.res['ind_sample_size'].clip(lower=4) - 3)
        else:
            self.res['ind_sample_size'] = np.nan
            self.res['var_ind'] = np.nan

        # Filter based on minimum sample size
        if self.full_rel_only:
            # Remove rows with insufficient samples
            self.res = self.res[self.res['pop_sample_size'] >= self.hyper['min_samples_needed']]
            self.res = self.res[self.res['ind_sample_size'] >= self.hyper['min_samples_needed']]
        else:
            # Set to NaN but keep rows
            mask_pop = (self.res['pop_sample_size'] < self.hyper['min_samples_needed']) | \
                       (self.res['pop_sample_size'].isna())
            self.res.loc[mask_pop, 'rel_pop'] = np.nan

            mask_ind = (self.res['ind_sample_size'] < self.hyper['min_samples_needed']) | \
                       (self.res['ind_sample_size'].isna())
            self.res.loc[mask_ind, 'rel_ind'] = np.nan

        # === Apply Fisher Transform ===
        self.res['rel_pop'] = fisher_transform(self.res['rel_pop'])
        self.res['rel_ind'] = fisher_transform(self.res['rel_ind'])

        # === Z-Normalize Correlations ===
        # Population
        valid_pop = self.res['rel_pop'].dropna()
        if not valid_pop.empty:
            mean_pop = valid_pop.mean()
            std_pop = valid_pop.std()
            if std_pop != 0 and not np.isnan(std_pop):
                self.res['rel_pop'] = (self.res['rel_pop'] - mean_pop) / std_pop
                # Scale variance accordingly
                self.res['var_pop'] = self.res['var_pop'] / std_pop

        # Individual
        valid_ind = self.res['rel_ind'].dropna()
        if not valid_ind.empty:
            mean_ind = valid_ind.mean()
            std_ind = valid_ind.std()
            if std_ind != 0 and not np.isnan(std_ind):
                self.res['rel_ind'] = (self.res['rel_ind'] - mean_ind) / std_ind
                # Scale variance accordingly
                self.res['var_ind'] = self.res['var_ind'] / std_ind

        # === Apply Sigmoid Transform ===
        self.res['rel_pop'] = self._apply_sigmoid(self.res['rel_pop'])
        self.res['rel_ind'] = self._apply_sigmoid(self.res['rel_ind'])

        # === Adjust Variances for Sigmoid ===
        # After sigmoid, variance = f'(x)^2 * var(x) where f'(x) = f(x) * (1 - f(x))
        self.res['var_pop'] *= self.res['rel_pop'] * (1 - self.res['rel_pop'])
        self.res['var_ind'] *= self.res['rel_ind'] * (1 - self.res['rel_ind'])

        # Convert standard errors to variances
        self.res['var_pop'] = self.res['var_pop'] ** 2
        self.res['var_ind'] = self.res['var_ind'] ** 2

        # Estimate prior variance as mean of population variance
        self.res['var_prior'] = self.res['var_pop'].dropna().mean()

    def _calculate_mi_features(self) -> None:
        """Calculate mutual information features (not yet implemented)."""
        # TODO: Implement MI-based feature calculation
        raise NotImplementedError("MI relationship type is not implemented yet")

    def _compute_pop_posterior(self) -> None:
        """
        Compute posterior distribution after population correlation update.

        Uses Bayesian update: Prior (LLM) + Likelihood (Population) → Posterior
        """
        # Prepare prior (LLM)
        llm_prior = self.res['llm_prior'].values
        llm_var = np.diag(self.hyper['alpha_prior'] * self.res['var_prior'].values)
        llm_var = np.nan_to_num(llm_var, nan=1e-8)

        # Prepare likelihood (Population)
        pop_r = self.res['rel_pop'].values
        pop_r = np.nan_to_num(pop_r, nan=1e-8)
        pop_var = np.diag(self.hyper['alpha_pop'] * self.res['var_pop'].values)
        pop_var = np.nan_to_num(pop_var, nan=1e-8)

        # Bayesian update
        mu_pop, sigma_pop = update_prior(llm_prior, llm_var, pop_r, pop_var)

        # Store posterior
        self.res['posterior_pop'] = mu_pop
        self.res['posterior_pop_var'] = np.diag(sigma_pop)

        # Mark insufficient samples as NaN
        mask = (self.res['pop_sample_size'] < self.hyper['min_samples_needed']) | \
               (self.res['pop_sample_size'].isna())
        self.res.loc[mask, 'posterior_pop'] = np.nan
        self.res.loc[mask, 'posterior_pop_var'] = np.nan
        self.res.loc[mask, 'rel_pop'] = np.nan

        # Store for individual update
        self._mu_pop = mu_pop
        self._sigma_pop = sigma_pop

    def _compute_ind_posterior(self) -> None:
        """
        Compute posterior distribution after individual correlation update.

        Uses Bayesian update: Posterior (Pop) + Likelihood (Individual) → Posterior (Ind)
        """
        if not hasattr(self, '_mu_pop') or not hasattr(self, '_sigma_pop'):
            raise ValueError("Population posterior must be computed before individual posterior.")

        # Prepare likelihood (Individual)
        ind_r = self.res['rel_ind'].values
        ind_r = np.nan_to_num(ind_r, nan=1e-8)
        ind_var = np.diag(self.hyper['alpha_ind'] * self.res['var_ind'].values)
        ind_var = np.nan_to_num(ind_var, nan=1e-8)

        # Bayesian update
        mu_ind, sigma_ind = update_prior(self._mu_pop, self._sigma_pop, ind_r, ind_var)

        # Store posterior
        self.res['posterior_ind'] = mu_ind
        self.res['posterior_ind_var'] = np.diag(sigma_ind)

        # Mark insufficient samples as NaN
        mask = (self.res['ind_sample_size'] < self.hyper['min_samples_needed']) | \
               (self.res['ind_sample_size'].isna())
        self.res.loc[mask, 'rel_ind'] = np.nan
        self.res.loc[mask, 'var_ind'] = np.nan
        self.res.loc[mask, 'posterior_ind'] = np.nan
        self.res.loc[mask, 'posterior_ind_var'] = np.nan

    def _handle_nan_posterior(self) -> None:
        """Fill NaN posteriors with fallback values (currently unused)."""
        # If posterior_pop is NaN, use llm_prior
        if 'posterior_pop' in self.res.columns:
            self.res.loc[pd.isna(self.res['posterior_pop']), 'posterior_pop'] = \
                self.res.loc[pd.isna(self.res['posterior_pop']), 'llm_prior']

        # If posterior_ind is NaN, use posterior_pop
        if 'posterior_ind' in self.res.columns:
            self.res.loc[pd.isna(self.res['posterior_ind']), 'posterior_ind'] = \
                self.res.loc[pd.isna(self.res['posterior_ind']), 'posterior_pop']

    def _final_weighting(self) -> None:
        """
        Compute final weights by combining global and local signals.

        Final weight = beta * local_weight + (1 - beta) * global_weight
        where:
        - local_weight: Based on recent abnormality patterns
        - global_weight: Based on Bayesian posterior (ind → pop → prior)
        """
        # Filter to data-associated metrics only
        self.res = self.res[self.res.index.isin(self.data_associated_metrics)]

        # Compute global weight with fallback chain: ind → pop → prior
        self.res['weight_global'] = self.res['posterior_ind'].where(
            ~self.res['posterior_ind'].isna(),
            self.res['posterior_pop'].where(
                ~self.res['posterior_pop'].isna(),
                self.res['llm_prior']
            )
        )

        # Combine global and local weights
        if self.ind_df is not None:
            # If local weight available, blend with global
            self.res['weight_final'] = self.hyper['beta'] * self.res['weight_local'].fillna(0) + \
                                        (1 - self.hyper['beta']) * self.res['weight_global']

            # If local weight is NaN for a metric, use only global
            self.res.loc[self.res['weight_local'].isna(), 'weight_final'] = \
                self.res.loc[self.res['weight_local'].isna(), 'weight_global']
        else:
            # No local information, use only global
            self.res['weight_final'] = self.res['weight_global']

    def _remove_query_metric(self) -> None:
        """Remove the query metric itself from results (avoid self-loops)."""
        self.res = self.res[self.res.index != self.query_metric]

    def update_rel_info(self, rel_info: Dict) -> None:
        """
        Update relationship information.

        Args:
            rel_info: Dictionary containing relationship statistics
        """
        # Population relationship matrices
        self.rel_pop_all = rel_info['rel_pop_all']
        if self.rel_pop_all is not None:
            # Compute global statistics (excluding diagonal)
            rel_pop_vals = self.rel_pop_all.values
            mask = ~np.eye(rel_pop_vals.shape[0], dtype=bool)
            off_diag_vals = rel_pop_vals[mask]
            if len(off_diag_vals) > 0:
                self.rel_pop_mean = np.nanmean(off_diag_vals)
                self.rel_pop_std = np.nanstd(off_diag_vals)
            else:
                self.rel_pop_mean = np.nan
                self.rel_pop_std = np.nan

        self.rel_var_pop_all = rel_info['rel_var_pop_all']
        self.rel_pop_sample_size_all = rel_info['rel_pop_sample_size_all']

        # Individual relationship matrices
        self.rel_ind_all = rel_info['rel_ind_all']
        if self.rel_ind_all is not None:
            rel_ind_vals = self.rel_ind_all.values
            mask = ~np.eye(rel_ind_vals.shape[0], dtype=bool)
            off_diag_vals = rel_ind_vals[mask]
            valid_vals = off_diag_vals[~np.isnan(off_diag_vals)]
            if len(valid_vals) > 0:
                self.rel_ind_mean = np.nanmean(valid_vals)
                self.rel_ind_std = np.nanstd(valid_vals)
            else:
                self.rel_ind_mean = np.nan
                self.rel_ind_std = np.nan

        self.rel_var_ind_all = rel_info['rel_var_ind_all']
        self.rel_ind_sample_size_all = rel_info['rel_ind_sample_size_all']

        # Metric lists
        self.numeric_metrics = rel_info['numeric_metrics']
        self.data_associated_metrics = rel_info['data_associated_metrics']

    def sort_by_column(self, column_name: str) -> pd.DataFrame:
        """
        Sort results by specified column in descending order.

        Args:
            column_name: Column name to sort by

        Returns:
            Sorted DataFrame
        """
        return self.res.sort_values(by=column_name, ascending=False)

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the complete weight computation pipeline.

        Returns:
            Tuple of (weights_dataframe, abnormality_dataframe)

        Workflow:
            1. Detect abnormalities in recent data
            2. Calculate correlation features
            3. Compute population posterior
            4. Compute individual posterior
            5. Remove query metric (avoid self-loop)
            6. Combine global and local weights
        """
        # Step 1: Handle abnormality detection
        self._handle_abnormality()

        # Store query metric's abnormality
        if self.ind_df is not None and self.query_metric in self.data_associated_metrics:
            self.query_abnormality = self.res.loc[self.query_metric, 'recent_abnormality']
        else:
            self.query_abnormality = np.nan

        # Step 2: Calculate relationship features
        if self.rel_type in ['spearman', 'pearson']:
            self._calculate_correlation_features()
        elif self.rel_type == 'mi':
            self._calculate_mi_features()
        else:
            raise ValueError(f"Invalid relation type: {self.rel_type}")

        # Step 3-4: Bayesian updates
        self._compute_pop_posterior()
        self._compute_ind_posterior()

        # Step 5: Remove query metric from results
        self._remove_query_metric()

        # Step 6: Create abnormality DataFrame
        self.abnormality_df = pd.DataFrame(index=self.res.index, columns=['abnormality'])
        if self.ind_df is not None:
            self.abnormality_df['abnormality'] = self.res['recent_abnormality']
            self.abnormality_df.loc[self.query_metric, 'abnormality'] = self.query_abnormality

        # Step 7: Compute final weights
        self._final_weighting()

        return self.res, self.abnormality_df
