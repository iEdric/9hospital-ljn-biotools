from typing import List, Tuple

import numpy as np


def calculate_bootstrap_ci(test_scores: List[float], n_bootstrap: int = 1000, ci_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for test scores.

    Args:
        test_scores: List of test scores (0.0 to 1.0 for each test)
        n_bootstrap: Number of bootstrap samples to generate
        ci_level: Confidence interval level (default: 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) representing the confidence interval
    """
    if not test_scores:
        return (0.0, 0.0)

    # Convert to numpy array for efficiency
    scores = np.array(test_scores)

    # Generate bootstrap samples
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))

    # Calculate confidence interval
    alpha = (1 - ci_level) / 2
    lower_bound = np.percentile(bootstrap_means, alpha * 100)
    upper_bound = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return (lower_bound, upper_bound)


def perform_permutation_test(scores_a: List[float], scores_b: List[float], n_permutations: int = 10000) -> Tuple[float, float]:
    """
    Perform a permutation test to determine if there's a significant difference
    between two sets of test scores.

    Args:
        scores_a: List of test scores for candidate A
        scores_b: List of test scores for candidate B
        n_permutations: Number of permutations to perform

    Returns:
        Tuple of (observed_difference, p_value)
    """
    if not scores_a or not scores_b:
        return (0.0, 1.0)

    # Calculate observed difference in means
    observed_diff = np.mean(scores_a) - np.mean(scores_b)

    # Combine all scores
    combined = np.concatenate([scores_a, scores_b])
    n_a = len(scores_a)

    # Perform permutation test
    count_greater_or_equal = 0
    for _ in range(n_permutations):
        # Shuffle the combined array
        np.random.shuffle(combined)

        # Split into two groups of original sizes
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]

        # Calculate difference in means
        perm_diff = np.mean(perm_a) - np.mean(perm_b)

        # Count how many permuted differences are >= to observed difference in absolute value
        if abs(perm_diff) >= abs(observed_diff):
            count_greater_or_equal += 1

    # Calculate p-value
    p_value = count_greater_or_equal / n_permutations

    return (observed_diff, p_value)
