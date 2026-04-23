#!/usr/bin/env python3
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
                                 verbose: bool = False) -> dict:
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
    
    # Calculate improvement
    improvement_pct = ((r_dense["d_prod"] - r_hadamard["d_prod"]) / r_dense["d_prod"]) * 100
    print(f"\n>>> Hadamard improvement: {improvement_pct:+.2f}%")
    
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
        "improvement_pct": improvement_pct,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate Hadamard rotation on real embeddings")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
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
    results = run_validation_on_real_data(vectors, data_source, args.verbose)
    
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
