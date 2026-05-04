"""
Statistical utility functions for retrieval package.

This module provides statistical operations including:
- Correlation computation (Pearson, Spearman)
- Mutual information estimation
- Fisher transformation
- Bayesian prior updates
- Anomaly detection
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, gaussian_kde, entropy
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import KernelDensity
import warnings
from typing import Tuple, Dict, Optional

from ..config import config


def fisher_transform(r: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Apply Fisher z-transformation to correlation coefficients.

    The Fisher transformation converts correlation coefficients to an
    approximately normal distribution, which is useful for inference.

    Args:
        r: Array of correlation coefficients (must be in [-1, 1])
        eps: Small constant to prevent log(0) errors

    Returns:
        Fisher-transformed values
    """
    r = np.clip(r, -1 + eps, 1 - eps)
    return 0.5 * np.log((1 + r) / (1 - r))


def z_normalize(x: np.ndarray, mean: Optional[float] = None, std: Optional[float] = None) -> np.ndarray:
    """
    Z-score normalization (standardization).

    Args:
        x: Array to normalize
        mean: Mean to use (if None, computed from x)
        std: Standard deviation to use (if None, computed from x)

    Returns:
        Z-normalized array
    """
    if mean is None:
        mean = np.mean(x)
    if std is None:
        std = np.std(x)
    if std == 0:
        return np.zeros_like(x)
    return (x - mean) / std


def min_max_normalize(x: np.ndarray) -> np.ndarray:
    """
    Min-max normalization to [0, 1] range.

    Args:
        x: Array to normalize

    Returns:
        Min-max normalized array
    """
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val == min_val:
        return np.ones_like(x) * 0.5
    return (x - min_val) / (max_val - min_val)


def freedman_diaconis_bins(data: np.ndarray) -> int:
    """
    Calculate optimal number of bins using Freedman-Diaconis rule.

    Args:
        data: Array of data points

    Returns:
        Optimal number of bins (clipped between 5 and 50)
    """
    data = data[~np.isnan(data)]
    if len(data) < 2:
        return 1

    data_range = np.max(data) - np.min(data)
    if data_range == 0:
        return 1

    # Freedman-Diaconis rule: h = 2 * IQR / n^(1/3)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    h = 2 * iqr / (len(data) ** (1 / 3))

    if h == 0:
        return 1

    bins = int(np.ceil(data_range / h))
    return np.clip(bins, *config.MUTUAL_INFO_BIN_RANGE)


def update_prior(
    mu0: np.ndarray,
    Sigma0: np.ndarray,
    r1: np.ndarray,
    V1: np.ndarray,
    reg_strength: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bayesian update of prior distribution with new observations.

    Updates a Gaussian prior N(mu0, Sigma0) with observations r1 ~ N(r1, V1)
    to produce posterior N(mu1, Sigma1) using conjugate Gaussian updates.

    Args:
        mu0: Prior mean vector
        Sigma0: Prior covariance matrix
        r1: Observed relationship vector
        V1: Observation covariance matrix
        reg_strength: Regularization strength for numerical stability

    Returns:
        Tuple of (posterior_mean, posterior_covariance)
    """
    # Use pseudo-inverse for numerical stability
    Sigma0_inv = np.linalg.pinv(Sigma0)
    V1_inv = np.linalg.pinv(V1)

    # Posterior covariance: (Sigma0^-1 + V1^-1)^-1
    Sigma1 = np.linalg.pinv(Sigma0_inv + V1_inv)

    # Posterior mean: Sigma1 @ (Sigma0^-1 @ mu0 + V1^-1 @ r1)
    mu1 = Sigma1 @ (Sigma0_inv @ mu0 + V1_inv @ r1)

    return mu1, Sigma1


def blend_openness_abnormality(
    x: np.ndarray,
    S: float,
    sharpness: float = 0.7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Blend abnormality scores with openness parameter using sigmoid.

    This function creates a weighted combination of forward and reverse-normalized
    abnormality scores, controlled by an openness parameter S:
    - S=1: Use original ranking (high abnormality = high weight)
    - S=0: Use reverse ranking (low abnormality = high weight)

    Args:
        x: Array of abnormality scores
        S: Openness parameter in [0, 1]
        sharpness: Temperature parameter for sigmoid transformation

    Returns:
        Tuple of (normalized, mixed, weighted) arrays
    """
    x = np.array(x)
    xmin, xmax = np.nanmin(x), np.nanmax(x)

    # 1) Normalize to [0, 1]
    if xmax == xmin:
        z = np.full_like(x, 0.5, dtype=float)
    else:
        z = (x - xmin) / (xmax - xmin)

    # 2) Mix forward and reverse based on openness S
    y = S * z + (1 - S) * (1 - z)

    # 3) Z-normalize (for applying sigmoid)
    y_mean = np.nanmean(y)
    y_std = np.nanstd(y)
    if y_std == 0:
        y2 = np.where(np.isnan(y), np.nan, 0.0)
    else:
        y2 = np.where(np.isnan(y), np.nan, (y - y_mean) / y_std)

    # 4) Apply sigmoid to map to [0, 1]
    y2 = np.where(np.isnan(y2), np.nan, 1 / (1 + np.exp(-sharpness * y2)))

    return z, y, y2


def compute_correlation_stats(
    df: pd.DataFrame,
    metrics: list,
    corr_type: str = 'spearman'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise correlations between all metrics.

    Args:
        df: DataFrame containing metric columns
        metrics: List of metric column names
        corr_type: Type of correlation ('spearman' or 'pearson')

    Returns:
        Tuple of (correlation_matrix, p_value_matrix, sample_size_matrix)
    """
    n_metrics = len(metrics)
    cor_matrix = np.zeros((n_metrics, n_metrics))
    p_matrix = np.zeros((n_metrics, n_metrics))
    sample_size_matrix = np.zeros((n_metrics, n_metrics))

    # Set diagonal to 1 (perfect self-correlation)
    np.fill_diagonal(cor_matrix, 1.0)
    np.fill_diagonal(p_matrix, 0.0)

    for i in range(n_metrics):
        for j in range(i + 1, n_metrics):
            metric1 = metrics[i]
            metric2 = metrics[j]

            # Get data for both metrics
            data1 = df[metric1].values
            data2 = df[metric2].values

            # Remove NaN values
            valid_mask = ~(np.isnan(data1) | np.isnan(data2))
            data1 = data1[valid_mask]
            data2 = data2[valid_mask]

            # Check if enough samples and not constant
            if len(data1) < 3 or np.std(data1) == 0 or np.std(data2) == 0:
                cor_matrix[i, j] = np.nan
                cor_matrix[j, i] = np.nan
                p_matrix[i, j] = np.nan
                p_matrix[j, i] = np.nan
            else:
                # Compute correlation
                if corr_type == 'spearman':
                    rho, p_val = spearmanr(data1, data2)
                elif corr_type == 'pearson':
                    rho, p_val = pearsonr(data1, data2)
                else:
                    raise ValueError(f"Unsupported correlation type: {corr_type}")

                cor_matrix[i, j] = rho
                cor_matrix[j, i] = rho
                p_matrix[i, j] = p_val
                p_matrix[j, i] = p_val

            sample_size_matrix[i, j] = len(data1)
            sample_size_matrix[j, i] = len(data1)

    # Create DataFrames
    cor_df = pd.DataFrame(cor_matrix, index=metrics, columns=metrics)
    p_df = pd.DataFrame(p_matrix, index=metrics, columns=metrics)
    sample_size_df = pd.DataFrame(sample_size_matrix, index=metrics, columns=metrics)

    return cor_df, p_df, sample_size_df


def estimate_mi_knn(x: np.ndarray, y: np.ndarray, n_neighbors: int = 10) -> float:
    """
    Estimate mutual information using k-nearest neighbors.

    Args:
        x: First variable
        y: Second variable
        n_neighbors: Number of neighbors for KNN estimator

    Returns:
        Estimated mutual information
    """
    x = x.reshape(-1, 1)
    return mutual_info_regression(x, y, n_neighbors=n_neighbors, random_state=config.RANDOM_SEED)[0]


def compute_mutual_info_stats(
    df: pd.DataFrame,
    metrics: list,
    bin_method: str = 'fd'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Compute pairwise mutual information between all metrics with bootstrapping.

    Args:
        df: DataFrame containing metric columns
        metrics: List of metric column names
        bin_method: Binning method ('fd', 'sqrt', or integer)

    Returns:
        Tuple of (mi_matrix, mi_var_matrix, sample_size_matrix, bootstrap_dict)
    """
    n_metrics = len(metrics)
    mi_matrix = np.ones((n_metrics, n_metrics))
    mi_var_matrix = np.ones((n_metrics, n_metrics))
    mi_sample_size_matrix = np.zeros((n_metrics, n_metrics))
    boot_matrix = {}

    for i in range(n_metrics):
        for j in range(i + 1, n_metrics):
            metric1 = metrics[i]
            metric2 = metrics[j]

            # Get data
            data1 = df[metric1].values
            data2 = df[metric2].values

            # Remove NaN values
            valid_mask = ~(np.isnan(data1) | np.isnan(data2))
            data1 = data1[valid_mask]
            data2 = data2[valid_mask]

            # Check if enough samples and not constant
            if len(data1) < 3 or np.std(data1) == 0 or np.std(data2) == 0:
                mi_matrix[i, j] = np.nan
                mi_matrix[j, i] = np.nan
                mi_var_matrix[i, j] = np.nan
                mi_var_matrix[j, i] = np.nan
                mi_sample_size_matrix[i, j] = len(data1)
                mi_sample_size_matrix[j, i] = len(data1)
                continue

            # Determine bins
            if isinstance(bin_method, int):
                bins = bin_method
            elif bin_method == 'fd':
                bins1 = freedman_diaconis_bins(data1)
                bins2 = freedman_diaconis_bins(data2)
                bins = max(bins1, bins2)
            elif bin_method == 'sqrt':
                bins = int(np.sqrt(len(data1)))
            else:
                raise ValueError(f"Invalid bin_method: {bin_method}")

            # Bootstrap MI estimation
            mi_list = []
            n_boot = config.MUTUAL_INFO_BOOTSTRAP_SAMPLES
            for _ in range(n_boot):
                min_len = len(data1)
                random_indices = np.random.choice(min_len, min_len)

                # Compute 2D histogram
                hist_2d, _, _ = np.histogram2d(data1[random_indices], data2[random_indices], bins=bins)
                mi_unnormalized = mutual_info_score(None, None, contingency=hist_2d)

                # Compute marginal entropies
                hist1, _ = np.histogram(data1, bins=bins)
                hist2, _ = np.histogram(data2, bins=bins)

                prob1 = hist1 / hist1.sum()
                prob2 = hist2 / hist2.sum()

                h1 = entropy(prob1, base=np.e)
                h2 = entropy(prob2, base=np.e)

                # Normalized mutual information
                normalized_mi = mi_unnormalized / min(h1, h2) if min(h1, h2) > 0 else 0
                mi_list.append(normalized_mi)

            # Store results
            mi = np.mean(mi_list)
            mi_var = np.var(mi_list)
            boot_matrix[(i, j)] = mi_list
            boot_matrix[(j, i)] = mi_list

            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi
            mi_var_matrix[i, j] = mi_var
            mi_var_matrix[j, i] = mi_var
            mi_sample_size_matrix[i, j] = len(data1)
            mi_sample_size_matrix[j, i] = len(data1)

    # Create DataFrames
    mi_df = pd.DataFrame(mi_matrix, index=metrics, columns=metrics)
    mi_var_df = pd.DataFrame(mi_var_matrix, index=metrics, columns=metrics)
    mi_sample_size_df = pd.DataFrame(mi_sample_size_matrix, index=metrics, columns=metrics)

    return mi_df, mi_var_df, mi_sample_size_df, boot_matrix


def detect_anomalies(
    df: pd.DataFrame,
    query_date: str,
    n_days: int
) -> Dict[str, float]:
    """
    Detect anomalies in the last n days ending on query_date.

    Compares recent data against historical patterns using z-scores.

    Args:
        df: DataFrame with 'date' column and metric columns
        query_date: End date for recent period (datetime string or object)
        n_days: Number of recent days to analyze

    Returns:
        Dictionary mapping metric names to mean absolute z-scores
    """
    if n_days == 'all':
        return {}

    temp_df = df.copy()
    temp_df['date'] = pd.to_datetime(temp_df['date'])
    query_date = pd.to_datetime(query_date)

    # Filter data up to query_date
    temp_df = temp_df[temp_df['date'] <= query_date].sort_values(by='date')

    if n_days == 'all' or len(temp_df) < n_days + 1:
        return {}

    # Split into recent and historical
    recent_data = temp_df.iloc[-n_days:]
    historical_data = temp_df.copy()

    anomaly_results = {}
    features = [col for col in temp_df.columns if col != 'date']

    for feature in features:
        if historical_data[feature].isna().all():
            anomaly_results[feature] = np.nan
            continue

        mean = historical_data[feature].mean()
        std = historical_data[feature].std()

        if std == 0 or np.isnan(std):
            anomaly_results[feature] = np.nan
        else:
            z_scores = (recent_data[feature] - mean) / std
            anomaly_scores = np.mean(np.abs(z_scores))
            anomaly_results[feature] = anomaly_scores

    return anomaly_results


def get_prior_matrix(root_path: str) -> pd.DataFrame:
    """
    Load LLM prior matrix from knowledge graph nodes and edges.

    Args:
        root_path: Path to directory containing nodes.json and edges.json

    Returns:
        DataFrame with LLM prior weights between all node pairs
    """
    import json
    from shared.models.entity import Entity
    from shared.models.relationship import Relationship

    # Load nodes
    with open(f"{root_path}/nodes.json", "r") as f:
        nodes = json.load(f)
    for node in nodes:
        nodes[node] = Entity.from_dict(nodes[node])

    # Load edges
    with open(f"{root_path}/edges.json", "r") as f:
        edges = json.load(f)
    for edge in edges:
        edges[edge] = Relationship.from_dict(edges[edge])

    # Create prior matrix
    name_list = [nodes[node].name for node in nodes]
    llm_prior_df = pd.DataFrame(
        data=np.zeros((len(name_list), len(name_list))),
        index=name_list,
        columns=name_list
    )

    # Set diagonal to 1 (self-relationship)
    llm_prior_df.loc[name_list, name_list] = 1

    # Fill in edge weights
    for edge in edges:
        edge_weight = edges[edge].weight
        llm_prior_df.loc[edges[edge].entity_1_name, edges[edge].entity_2_name] = float(edge_weight)
        llm_prior_df.loc[edges[edge].entity_2_name, edges[edge].entity_1_name] = float(edge_weight)

    return llm_prior_df


def get_relationship_matrix(
    relationship_dict: Dict,
    rel_type: str = 'spearman',
    data_type: str = 'pop'
) -> Tuple:
    """
    Extract relationship matrices from relationship dictionary.

    Args:
        relationship_dict: Dictionary containing relationship statistics
        rel_type: Relationship type ('spearman', 'pearson', 'mi')
        data_type: Data type ('pop' or 'ind')

    Returns:
        Tuple of (rel_matrix, sample_size_matrix, p_or_var_matrix, metrics_list)
    """
    key = f'{rel_type}_{data_type}'
    temp = pd.DataFrame(relationship_dict[key])
    numeric_metrics = list(temp.columns)

    rel_all = pd.DataFrame(
        relationship_dict[key],
        index=numeric_metrics,
        columns=numeric_metrics
    )
    sample_size_all = pd.DataFrame(
        relationship_dict[f'sample_size_{data_type}'],
        index=numeric_metrics,
        columns=numeric_metrics
    )

    if rel_type == 'mi':
        var_all = pd.DataFrame(
            relationship_dict[f'{rel_type}_var_{data_type}'],
            index=numeric_metrics,
            columns=numeric_metrics
        )
        return rel_all, sample_size_all, var_all, numeric_metrics
    else:
        p_all = pd.DataFrame(
            relationship_dict[f'{rel_type}_p_{data_type}'],
            index=numeric_metrics,
            columns=numeric_metrics
        )
        return rel_all, sample_size_all, p_all, numeric_metrics
