"""
Evaluation Utilities Module

This module provides statistical analysis and reporting utilities for evaluation results.
It includes functions for calculating win rates, generating LaTeX tables, and analyzing
evaluation metrics across different dimensions.

Key Features:
    - Win rate calculation and statistical analysis
    - Grouped analysis by query type, dataset, difficulty, etc.
    - LaTeX table generation for academic papers
    - Distribution analysis and reporting

Functions:
    - calculate_wr: Calculate overall win rates for methods
    - calculate_wr_by: Calculate win rates grouped by a specific key
    - generate_latex_table: Generate publication-ready LaTeX tables
    - count_per_type: Count and format results by category type

Usage Example:
    >>> import pandas as pd
    >>> from evaluation.utils import calculate_wr, generate_latex_table
    >>>
    >>> # Load evaluation results
    >>> results_df = pd.DataFrame([
    ...     {'q_id': 1, 'method': 'baseline', 'overall_quality': 2.0},
    ...     {'q_id': 1, 'method': 'rag', 'overall_quality': 1.5},
    ...     {'q_id': 1, 'method': 'kgrag', 'overall_quality': 1.0},
    ...     # ... more results
    ... ])
    >>>
    >>> # Calculate win rates
    >>> win_rates = calculate_wr(results_df)
    >>> print(win_rates)
    >>>
    >>> # Generate LaTeX table
    >>> mean_scores = results_df.groupby('method')['overall_quality'].mean()
    >>> generate_latex_table(mean_scores, win_rates,
    ...                      caption="Evaluation Results",
    ...                      label="tab:results")

"""

from typing import List, Dict, Optional, Union
import logging

import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


def calculate_wr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate win rates for each method based on overall quality rankings.

    A "win" is defined as achieving rank 1 (best quality score) for a given query.
    Lower quality scores indicate better performance. This function counts how many
    times each method achieves the best rank across all queries.

    Args:
        df: DataFrame with columns:
            - q_id: Query identifier
            - method: Method name (e.g., 'baseline', 'rag', 'kgrag')
            - overall_quality: Quality score (lower is better, 1.0 = best)

    Returns:
        DataFrame with columns:
            - index: Method name
            - wins: Number of times method achieved rank 1
            - win_rate: Proportion of queries where method won

    Example:
        >>> results = pd.DataFrame([
        ...     {'q_id': 1, 'method': 'baseline', 'overall_quality': 2.0},
        ...     {'q_id': 1, 'method': 'rag', 'overall_quality': 3.0},
        ...     {'q_id': 1, 'method': 'kgrag', 'overall_quality': 1.0},
        ...     {'q_id': 2, 'method': 'baseline', 'overall_quality': 1.0},
        ...     {'q_id': 2, 'method': 'rag', 'overall_quality': 2.0},
        ...     {'q_id': 2, 'method': 'kgrag', 'overall_quality': 3.0},
        ... ])
        >>> win_rate_df = calculate_wr(results)
        >>> print(win_rate_df)
        #          wins  win_rate
        # baseline    1       0.5
        # rag         0       0.0
        # kgrag       1       0.5
    """
    logger.info("Calculating win rates for evaluation results")

    # Get unique methods
    methods = df['method'].unique()
    logger.debug(f"Found methods: {list(methods)}")

    # Initialize win counts
    win_counts = {method: 0 for method in methods}
    ranks = {method: [] for method in methods}

    valid_question_count = 0

    # Count wins per query
    for q_id, group in df.groupby('q_id'):
        # Skip queries with missing scores
        if group['overall_quality'].isna().any():
            logger.warning(f"Skipping query {q_id}: contains NaN scores")
            continue

        valid_question_count += 1

        # Collect quality scores (rank) for each method
        for method, rank in zip(group['method'], group['overall_quality']):
            ranks[method].append(rank)

    # Count wins (rank == 1.0)
    for method in ranks:
        win_counts[method] = ranks[method].count(1.0)

    # Calculate win rates
    logger.info(f"Processed {valid_question_count} valid queries")

    win_rate_df = pd.DataFrame.from_dict(
        win_counts,
        orient='index',
        columns=['wins']
    )
    denominator = valid_question_count if valid_question_count > 0 else 1
    win_rate_df['win_rate'] = win_rate_df['wins'] / denominator

    logger.info(f"Win rates calculated: {win_rate_df.to_dict()}")
    return win_rate_df


def calculate_wr_by(
    df: pd.DataFrame,
    key: str
) -> pd.DataFrame:
    """
    Calculate win rates grouped by a specific key (e.g., query type, dataset).

    This function performs win rate analysis within each category defined by the
    grouping key, allowing you to see how methods perform across different
    query types, datasets, difficulty levels, etc.

    Args:
        df: DataFrame with columns:
            - q_id: Query identifier
            - method: Method name
            - overall_quality: Quality score (lower is better)
            - {key}: Grouping variable (e.g., 'query_type', 'dataset')
        key: Column name to group by

    Returns:
        DataFrame with columns:
            - {key}: The grouping variable value
            - method: Method name
            - win_rate: Win rate within that group

    Example:
        >>> results = pd.DataFrame([
        ...     {'q_id': 1, 'method': 'baseline', 'overall_quality': 2.0, 'query_type': 'factual'},
        ...     {'q_id': 1, 'method': 'rag', 'overall_quality': 3.0, 'query_type': 'factual'},
        ...     {'q_id': 1, 'method': 'kgrag', 'overall_quality': 1.0, 'query_type': 'factual'},
        ...     {'q_id': 2, 'method': 'baseline', 'overall_quality': 1.0, 'query_type': 'analytical'},
        ...     {'q_id': 2, 'method': 'rag', 'overall_quality': 2.0, 'query_type': 'analytical'},
        ...     {'q_id': 2, 'method': 'kgrag', 'overall_quality': 3.0, 'query_type': 'analytical'},
        ... ])
        >>> grouped_wr = calculate_wr_by(results, 'query_type')
        >>> print(grouped_wr)
    """
    logger.info(f"Calculating win rates grouped by: {key}")

    if key not in df.columns:
        raise ValueError(f"Key '{key}' not found in DataFrame columns: {df.columns.tolist()}")

    methods = df['method'].unique()
    keys = df[key].unique()
    results = []

    logger.debug(f"Found {len(keys)} unique values for '{key}': {list(keys)}")

    for k in keys:
        # Filter data for this category
        query_df = df[df[key] == k]
        total_questions = len(query_df['q_id'].unique())

        if total_questions == 0:
            logger.warning(f"No queries found for {key}={k}")
            continue

        logger.debug(f"Processing {key}={k} with {total_questions} queries")

        # Initialize ranks
        ranks = {method: [] for method in methods}

        valid_question_count = 0

        # Count wins per question within this category
        for q_id, group in query_df.groupby('q_id'):
            if group['overall_quality'].isna().any():
                logger.warning(f"Skipping query {q_id} in {key}={k}: contains NaN scores")
                continue

            valid_question_count += 1

            for method, rank in zip(group['method'], group['overall_quality']):
                ranks[method].append(rank)

        # Calculate win counts
        win_counts = {method: ranks[method].count(1.0) for method in ranks}
        logger.debug(f"{key}={k}: win counts = {win_counts}")

        # Calculate win rates for each method
        for method in methods:
            results.append({
                key: k,
                'method': method,
                'win_rate': win_counts[method] / valid_question_count if valid_question_count > 0 else 0
            })

    result_df = pd.DataFrame(results)
    logger.info(f"Grouped win rates calculated for {len(results)} combinations")
    return result_df


def generate_latex_table(
    mean_df: pd.DataFrame,
    win_rate_df: pd.DataFrame,
    caption: str,
    label: str
) -> None:
    """
    Generate a publication-ready LaTeX table with mean scores and win rates.

    This function creates a formatted LaTeX table suitable for academic papers,
    with automatic bolding of best results (lowest mean, highest win rate).
    The output is printed to stdout and can be redirected to a file.

    Args:
        mean_df: DataFrame with mean scores, indexed by dataset/category,
                columns are methods
        win_rate_df: DataFrame with win rates, same structure as mean_df
        caption: Table caption for LaTeX
        label: Table label for LaTeX cross-referencing

    Raises:
        ValueError: If DataFrames have different shapes or columns

    Example:
        >>> import pandas as pd
        >>> mean_df = pd.DataFrame({
        ...     'baseline': [2.5, 2.3],
        ...     'rag': [2.1, 2.0],
        ...     'kgrag': [1.8, 1.7]
        ... }, index=['dataset_a', 'dataset_b'])
        >>> win_rate_df = pd.DataFrame({
        ...     'baseline': [0.2, 0.25],
        ...     'rag': [0.3, 0.35],
        ...     'kgrag': [0.5, 0.4]
        ... }, index=['dataset_a', 'dataset_b'])
        >>> generate_latex_table(mean_df, win_rate_df,
        ...                      caption="Evaluation Results",
        ...                      label="tab:results")
        # Prints LaTeX table to stdout
    """
    logger.info(f"Generating LaTeX table: {label}")

    # Validation
    if mean_df.shape != win_rate_df.shape:
        raise ValueError(
            f"DataFrames must have the same shape. "
            f"mean_df: {mean_df.shape}, win_rate_df: {win_rate_df.shape}"
        )

    if not all(mean_df.columns == win_rate_df.columns):
        raise ValueError(
            f"DataFrames must have the same columns. "
            f"mean_df: {mean_df.columns.tolist()}, "
            f"win_rate_df: {win_rate_df.columns.tolist()}"
        )

    methods = mean_df.columns
    num_methods = len(methods)
    logger.debug(f"Generating table for {num_methods} methods: {list(methods)}")

    # Column format with vertical separators between method groups
    col_format = 'l|' + '|'.join(['cc'] * num_methods)

    # Print LaTeX table header
    print(f"\\begin{{table*}}[htbp]")
    print(f"\\centering")
    print(f"\\caption{{{caption}}}")
    print(f"\\label{{{label}}}")
    print(f"\\begin{{tabular}}{{{col_format}}}")
    print("\\toprule")

    # Method headers with multicolumn
    method_headers = [
        f'\\multicolumn{{2}}{{c|}}{{{method}}}'
        for method in methods[:-1]
    ] + [f'\\multicolumn{{2}}{{c}}{{{methods[-1]}}}']
    print(f"Dataset & {' & '.join(method_headers)} \\\\")
    print(f"\\cmidrule(lr){{2-{2*num_methods+1}}}")

    # Column headers (Mean, Win Rate)
    column_headers = ' & '.join(['Mean', 'Win Rate'] * num_methods)
    print(f" & {column_headers} \\\\")
    print("\\midrule")

    # Data rows
    for dataset in mean_df.index:
        # Find min mean and max win rate for this dataset
        min_mean = mean_df.loc[dataset].min()
        max_win_rate = win_rate_df.loc[dataset].max()

        row_values = []
        for method in methods:
            # Format mean value (bold if minimum)
            mean_val = mean_df.loc[dataset, method]
            if abs(mean_val - min_mean) < 0.001:  # Handle floating point comparison
                mean_str = f"\\textbf{{{mean_val:.2f}}}"
            else:
                mean_str = f"{mean_val:.2f}"

            # Format win rate (bold if maximum)
            win_rate_val = win_rate_df.loc[dataset, method]
            if abs(win_rate_val - max_win_rate) < 0.001:  # Handle floating point comparison
                win_rate_str = f"\\textbf{{{win_rate_val:.2f}}}"
            else:
                win_rate_str = f"{win_rate_val:.2f}"

            row_values.extend([mean_str, win_rate_str])

        print(f"{dataset} & {' & '.join(row_values)} \\\\")

    # Calculate and print averages if multiple datasets
    if len(mean_df) > 1:
        print("\\midrule")

        # Calculate overall averages
        avg_means = mean_df.mean()
        avg_win_rates = win_rate_df.mean()

        # Find min avg mean and max avg win rate
        min_avg_mean = avg_means.min()
        max_avg_win_rate = avg_win_rates.max()

        avg_values = []
        for method in methods:
            # Format average mean (bold if minimum)
            avg_mean = avg_means[method]
            if abs(avg_mean - min_avg_mean) < 0.001:
                avg_mean_str = f"\\textbf{{{avg_mean:.2f}}}"
            else:
                avg_mean_str = f"{avg_mean:.2f}"

            # Format average win rate (bold if maximum)
            avg_win_rate = avg_win_rates[method]
            if abs(avg_win_rate - max_avg_win_rate) < 0.001:
                avg_win_rate_str = f"\\textbf{{{avg_win_rate:.2f}}}"
            else:
                avg_win_rate_str = f"{avg_win_rate:.2f}"

            avg_values.extend([avg_mean_str, avg_win_rate_str])

        print(f"Average & {' & '.join(avg_values)} \\\\")

    # Print table footer
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table*}")

    logger.info("LaTeX table generated successfully")


def count_per_type(
    combined_df: pd.DataFrame,
    key: str,
    num_of_method: int,
    order: List[Union[str, int]]
) -> None:
    """
    Count and format query distribution by type, generating LaTeX tables.

    This function analyzes the distribution of queries across different categories
    (e.g., query types, difficulty levels) and generates LaTeX tables for reporting.
    It accounts for multiple methods by dividing counts appropriately.

    Args:
        combined_df: DataFrame containing evaluation results with:
            - dataset: Dataset name
            - {key}: Category to count by
            - Other columns...
        key: Column name to count by (e.g., 'query_type', 'difficulty')
        num_of_method: Number of methods evaluated (for count normalization)
        order: List defining the desired order of categories in output

    Example:
        >>> results = pd.DataFrame([
        ...     {'dataset': 'wearable_a', 'query_type': 'factual', 'method': 'baseline'},
        ...     {'dataset': 'wearable_a', 'query_type': 'factual', 'method': 'rag'},
        ...     {'dataset': 'wearable_a', 'query_type': 'factual', 'method': 'kgrag'},
        ...     {'dataset': 'wearable_a', 'query_type': 'analytical', 'method': 'baseline'},
        ...     # ... more rows
        ... ])
        >>> count_per_type(results, 'query_type', 3, ['factual', 'analytical'])
        # Prints LaTeX tables for each dataset
    """
    logger.info(f"Counting query distribution by '{key}'")

    if key not in combined_df.columns:
        raise ValueError(f"Key '{key}' not found in DataFrame columns")

    dataset_list = combined_df['dataset'].unique()
    logger.debug(f"Found {len(dataset_list)} datasets: {list(dataset_list)}")

    # Process each dataset
    for dataset in dataset_list:
        logger.info(f"Processing dataset: {dataset}")
        print(f"\n{dataset} {'>'*20}")

        # Get counts for this dataset (divided by number of methods)
        dataset_df = combined_df[combined_df['dataset'] == dataset]
        query_counts = (dataset_df[key].value_counts() / num_of_method).astype(int)

        # Create ordered DataFrame
        counts_df = (query_counts
                    .reindex(order)  # Ensure all categories are present
                    .fillna(0)       # Fill missing with 0
                    .reset_index())
        counts_df.columns = [key, dataset]

        logger.debug(f"Counts for {dataset}: {counts_df[dataset].tolist()}")

        # Transpose while maintaining order
        transposed_df = (counts_df
                        .set_index(key)
                        .reindex(order)  # Maintain order after transpose
                        .T
                        .reset_index())

        # Generate LaTeX table
        latex_table = transposed_df.to_latex(
            index=False,
            header=False,
            caption=f"Query counts for {dataset} (divided by {num_of_method})",
            label=f"tab:{dataset.lower()}_counts",
            position="htbp",
            escape=False
        )

        print(latex_table)

    # Generate total counts table
    logger.info("Generating total counts table")
    print("\nTotal Counts >>>>>>>>>>>>>>>>")

    total_counts = (combined_df[key].value_counts() / num_of_method).astype(int)
    total_df = (total_counts
                .reindex(order)
                .fillna(0)
                .reset_index())
    total_df.columns = ['Type', 'Total']

    logger.debug(f"Total counts: {total_df['Total'].tolist()}")

    total_transposed = (total_df
                       .set_index('Type')
                       .reindex(order)
                       .T
                       .reset_index())

    # Generate LaTeX table for totals
    total_latex = total_transposed.to_latex(
        index=False,
        header=False,
        caption=f"Total query counts (divided by {num_of_method})",
        label="tab:total_counts",
        position="htbp",
        escape=False
    )

    # Format header row
    order_str = [str(c) for c in order]
    header_row = " & " + " & ".join(order_str) + " \\\\"

    print(header_row)
    print(total_latex)

    logger.info("Count analysis completed")
