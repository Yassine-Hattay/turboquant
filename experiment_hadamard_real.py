#!/usr/bin/env python3
# experiment_hadamard_real.py
"""
Validate Hadamard rotation on real embeddings.

This script validates the Hadamard vs. Dense rotation comparison on real LLM embedding data,
reusing the validated logic from experiment2_fixed_rotation.py.
"""
import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cpu")

# Import validated functions from experiment2_fixed_rotation
from experiment2_fixed_rotation import (
    dense_rotation,
    hadamard_rotation,
    compute_d_prod_manual,
    assert_orthogonality,
)


def compute_channel_variance(X: torch.Tensor) -> torch.Tensor:
    """Compute per-channel variance across samples."""
    return X.var(dim=0)


def split_outlier_indices(var: torch.Tensor, outlier_ratio: float = 0.05):
    """Return (outlier_idx, regular_idx) tensors based on top variance."""
    d = var.shape[0]
    k = max(1, int(d * outlier_ratio))
    sorted_idx = torch.argsort(var, descending=True)
    return sorted_idx[:k], sorted_idx[k:]


def compute_d_prod_outlier_aware(
    X: torch.Tensor, Y: torch.Tensor, Pi_reg: torch.Tensor,
    boundaries: torch.Tensor, centroids: torch.Tensor, outlier_ratio: float = 0.05,
    X_raw_for_detection: torch.Tensor = None
) -> dict:
    """
    Outlier-aware D_prod for real embeddings:
    - Splits channels into outliers (no rotation, NO quantization) and regular (Hadamard rotated)
    - Outliers pass through unchanged; regular channels are rotated + quantized
    - Reconstructs full vector and computes D_prod, RMSE, correlation
    """
    X_norms = X.norm(dim=1, keepdim=True)
    X_unit = X / (X_norms + 1e-10)
    
    # FIX 1: Detect outliers on pre-normalized data to preserve variance structure
    if X_raw_for_detection is None:
        X_raw_for_detection = X * (X_norms + 1e-10)
    var = compute_channel_variance(X_raw_for_detection)
    outlier_idx, regular_idx = split_outlier_indices(var, outlier_ratio)
    
    # FIX 2: Do NOT quantize outliers - pass them through unchanged
    X_out_hat = X_unit[:, outlier_idx]
    
    # Rotate + quantize regular channels
    X_reg = X_unit[:, regular_idx]
    Y_reg = X_reg @ Pi_reg.T
    int_bounds = boundaries[1:-1]
    idx_reg = torch.searchsorted(int_bounds, Y_reg)
    X_reg_hat = centroids[idx_reg] @ Pi_reg
    
    # Reconstruct
    X_recon_unit = torch.zeros_like(X_unit)
    X_recon_unit[:, outlier_idx] = X_out_hat
    X_recon_unit[:, regular_idx] = X_reg_hat
    X_recon = X_recon_unit * X_norms
    
    ip_true = (X * Y).sum(dim=1)
    ip_quant = (X_recon * Y).sum(dim=1)
    d_prod = ((ip_true - ip_quant) ** 2).mean().item()
    true_ip_var = (ip_true ** 2).mean().item()
    relative_rmse = np.sqrt(d_prod / max(true_ip_var, 1e-10))
    correlation = torch.corrcoef(torch.stack([ip_true, ip_quant]))[0, 1].item()
    
    return {
        "d_prod": d_prod, "relative_rmse": relative_rmse, "correlation": correlation,
        "outlier_ratio": outlier_ratio, "n_outliers": len(outlier_idx), "n_regular": len(regular_idx)
    }


def load_real_embeddings():
    """Load real embeddings from highest-priority available file."""
    priority_files = [
        "real_embeddings_mteb.npy",
        "real_embeddings_minilm.npy",
        "real_embeddings_anisotropic.npy",
    ]
    
    for filename in priority_files:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        if os.path.exists(filepath):
            vectors = np.load(filepath)
            print(f"✓ Loaded {filename}: {vectors.shape[0]} vectors, dim={vectors.shape[1]}")
            return vectors, filename
    
    raise FileNotFoundError(
        "No real embedding files found. Run load_real_embeddings.py first."
    )


def get_codebook_for_dimension(dim: int):
    """Load codebook tensors for the given dimension."""
    codebook_map = {
        64: "codebook_d64_b3.json",
        128: "codebook_d128_b3.json",
        384: "codebook_d384_b3.json",
        576: "codebook_d576_b3.json",
    }
    
    if dim not in codebook_map:
        raise ValueError(f"No codebook available for dimension {dim}. Available: {list(codebook_map.keys())}")
    
    codebook_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f"turboquant/codebooks/{codebook_map[dim]}")
    
    with open(codebook_path, 'r') as f:
        codebook_data = json.load(f)
    
    boundaries = torch.tensor(codebook_data["boundaries"], device=device, dtype=torch.float32)
    centroids = torch.tensor(codebook_data["centroids"], device=device, dtype=torch.float32)
    
    return boundaries, centroids


def run_validation_on_real_data(vectors: np.ndarray, data_source: str, 
                                 verbose: bool = False, outlier_ratio: float = 0.05) -> dict:
    """Run Hadamard vs. Dense validation on real embeddings."""
    n_samples, d = vectors.shape
    print(f"\nReal data variance ratio: {np.var(vectors, axis=0).max() / (np.var(vectors, axis=0).min() + 1e-10):.2f}x")
    
    # Load matching codebook
    boundaries, centroids = get_codebook_for_dimension(d)
    if verbose:
        print(f"  Loaded codebook: {len(centroids)} centroids, {len(boundaries)} boundaries")
    
    # Convert to torch and normalize to unit sphere
    X_all = torch.tensor(vectors, device=device, dtype=torch.float32)
    X_all = X_all / (X_all.norm(dim=1, keepdim=True) + 1e-10)
    
    # Split into X (keys) and Y (queries)
    X = X_all[:n_samples//2]
    Y = X_all[n_samples//2:]
    
    # Generate rotation matrices with fixed seeds for reproducibility
    seed = 42
    hadamard_seed = 1337
    
    Pi_dense = dense_rotation(d, device, seed=seed)
    Pi_hadamard = hadamard_rotation(d, device, seed=hadamard_seed)
    
    # Verify orthogonality
    print("\n--- DENSE ---")
    dense_orth_error = torch.max(torch.abs(Pi_dense @ Pi_dense.T - torch.eye(d, device=device))).item()
    print(f"  Orthogonality error: {dense_orth_error:.2e}", end="")
    if dense_orth_error < 1e-4:
        print(" OK")
    else:
        print(" FAILED")
    
    r_dense = compute_d_prod_manual(X, Y, Pi_dense, boundaries, centroids)
    print(f"  D_prod: {r_dense['d_prod']:.6e}, Correlation: {r_dense['correlation']:.4f}")
    
    print("\n--- HADAMARD ---")
    hadamard_orth_error = torch.max(torch.abs(Pi_hadamard @ Pi_hadamard.T - torch.eye(d, device=device))).item()
    print(f"  Orthogonality error: {hadamard_orth_error:.2e}", end="")
    if hadamard_orth_error < 1e-4:
        print(" OK")
    else:
        print(" FAILED")
    
    r_hadamard = compute_d_prod_manual(X, Y, Pi_hadamard, boundaries, centroids)
    print(f"  D_prod: {r_hadamard['d_prod']:.6e}, Correlation: {r_hadamard['correlation']:.4f}")
    
    # Outlier-Aware Hadamard
    print("\n--- OUTLIER-AWARE HADAMARD ---")
    # Detect outliers on pre-normalized data (before unit sphere normalization)
    X_pre_norm = X_all[:n_samples//2] * (X_all[:n_samples//2].norm(dim=1, keepdim=True) + 1e-10)
    var = compute_channel_variance(X_pre_norm)
    outlier_idx, regular_idx = split_outlier_indices(var, outlier_ratio)
    Pi_reg = hadamard_rotation(len(regular_idx), device, seed=hadamard_seed)
    r_outlier = compute_d_prod_outlier_aware(X, Y, Pi_reg, boundaries, centroids, outlier_ratio, X_raw_for_detection=X_pre_norm)
    print(f"  Outliers: {r_outlier['n_outliers']}, Regular: {r_outlier['n_regular']}")
    print(f"  D_prod: {r_outlier['d_prod']:.6e}, Relative RMSE: {r_outlier['relative_rmse']:.4f}, Correlation: {r_outlier['correlation']:.4f}")
    
    # Calculate improvement
    improvement_pct = ((r_dense["d_prod"] - r_hadamard["d_prod"]) / r_dense["d_prod"]) * 100
    print(f"\n>>> Hadamard improvement: {improvement_pct:+.2f}%")
    
    # Print comparison table
    print("\n" + "=" * 70)
    print(f"{'Method':<25} {'D_prod':<18} {'Relative RMSE':<15} {'Correlation':<12}")
    print("=" * 70)
    print(f"{'Dense':<25} {r_dense['d_prod']:<18.6e} {'N/A':<15} {r_dense['correlation']:<12.4f}")
    print(f"{'Hadamard':<25} {r_hadamard['d_prod']:<18.6e} {'N/A':<15} {r_hadamard['correlation']:<12.4f}")
    print(f"{'Outlier-Aware Hadamard':<25} {r_outlier['d_prod']:<18.6e} {r_outlier['relative_rmse']:<15.4f} {r_outlier['correlation']:<12.4f}")
    print("=" * 70)
    
    return {
        "data_source": data_source,
        "dimension": d,
        "variance_ratio": float(np.var(vectors, axis=0).max() / (np.var(vectors, axis=0).min() + 1e-10)),
        "dense": {
            "orth_error": dense_orth_error,
            "d_prod": r_dense["d_prod"],
            "correlation": r_dense["correlation"],
        },
        "hadamard": {
            "orth_error": hadamard_orth_error,
            "d_prod": r_hadamard["d_prod"],
            "correlation": r_hadamard["correlation"],
        },
        "outlier_aware": {
            "d_prod": r_outlier["d_prod"],
            "relative_rmse": r_outlier["relative_rmse"],
            "correlation": r_outlier["correlation"],
            "outlier_ratio": r_outlier["outlier_ratio"],
            "n_outliers": r_outlier["n_outliers"],
            "n_regular": r_outlier["n_regular"],
        },
        "improvement_pct": improvement_pct,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate Hadamard rotation on real embeddings")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--outlier-ratio", type=float, default=0.05, 
                        help="Ratio of outlier channels (default: 0.05)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Hadamard Rotation Validation on Real Embeddings")
    print("=" * 60)
    
    # Load real embeddings
    try:
        vectors, data_source = load_real_embeddings()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'python load_real_embeddings.py' first to generate embeddings.")
        sys.exit(1)
    
    # Run validation
    results = run_validation_on_real_data(vectors, data_source, args.verbose, args.outlier_ratio)
    
    # Save results to JSON
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "real_data_hadamard_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\n✓ Saved to {results_path}")
    
    # Final status
    if results["improvement_pct"] > 3.0:
        print(f"\nSUCCESS: Hadamard on real data shows +{results['improvement_pct']:.2f}% improvement")
    elif results["improvement_pct"] > 0:
        print(f"\nPARTIAL: Hadamard shows modest +{results['improvement_pct']:.2f}% improvement")
    else:
        print(f"\nWARNING: Hadamard shows no improvement on real data ({results['improvement_pct']:.2f}%)")


if __name__ == "__main__":
    main()
