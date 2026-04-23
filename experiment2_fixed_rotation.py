#!/usr/bin/env python3
"""
Experiment 2 Fixed: Manual D_prod computation bypassing broken rotation injection.

The existing experiment2_structured_rotation.py tries to inject custom rotation matrices
via quantizer.Pi = Pi. Because Pi is registered with self.register_buffer() in TurboQuantMSE,
direct attribute assignment is ignored. This script computes D_prod manually to bypass
the broken injection entirely.
"""

import os
import sys
import json
import torch
import numpy as np

# Setup: Import from local turboquant directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# CPU-only, deterministic seeds
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cpu")


def dense_rotation(d: int, device: torch.device) -> torch.Tensor:
    """Generate dense random rotation: randn → QR → fix signs → return Q."""
    A = torch.randn(d, d, device=device, dtype=torch.float32)
    Q, R = torch.linalg.qr(A)
    # Fix signs: ensure diagonal of R is positive
    signs = torch.sign(torch.diag(R))
    Q = Q * signs.unsqueeze(0)
    return Q


def hadamard_rotation(d: int, device: torch.device) -> torch.Tensor:
    """
    Generate Hadamard rotation with orthogonality fix.
    
    CRITICAL FIX: Apply torch.linalg.qr to the truncated matrix and fix signs
    to restore orthogonality after truncation.
    """
    # Find next power of 2 >= d
    n = 1
    while n < d:
        n *= 2
    
    # Build Sylvester Hadamard matrix recursively
    def build_hadamard(size):
        if size == 1:
            return torch.tensor([[1.0]], device=device)
        H_prev = build_hadamard(size // 2)
        H_top = torch.cat([H_prev, H_prev], dim=1)
        H_bottom = torch.cat([H_prev, -H_prev], dim=1)
        return torch.cat([H_top, H_bottom], dim=0)
    
    H = build_hadamard(n)
    
    # Apply random Rademacher sign flips
    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)
    signs = (torch.randint(0, 2, (n,), generator=rng, device=device, dtype=torch.float32) * 2 - 1)
    H = H * signs.unsqueeze(0)
    
    # Truncate to d×d
    H_truncated = H[:d, :d]
    
    # CRITICAL FIX: Apply QR to restore orthogonality after truncation
    Q, R = torch.linalg.qr(H_truncated)
    # Fix signs
    diag_signs = torch.sign(torch.diag(R))
    Q = Q * diag_signs.unsqueeze(0)
    
    return Q


def assert_orthogonality(Pi: torch.Tensor, name: str) -> None:
    """Assert orthogonality: max|Pi @ Pi.T - I| < 1e-4. Fail immediately if violated."""
    d = Pi.shape[0]
    I = torch.eye(d, device=Pi.device)
    error = torch.max(torch.abs(Pi @ Pi.T - I)).item()
    if error >= 1e-4:
        raise AssertionError(f"{name} rotation failed orthogonality check: error={error:.6f}")
    print(f"  {name} orthogonality error: {error:.6f}")


def compute_d_prod_manual(X: torch.Tensor, Y: torch.Tensor, Pi: torch.Tensor, 
                          boundaries: torch.Tensor, centroids: torch.Tensor) -> dict:
    """
    Manual asymmetric quantization & D_prod computation.
    
    For a given rotation Pi:
    - Normalize X to unit vectors
    - Rotate: Y_rot = X_unit @ Pi.T
    - Quantize: indices = searchsorted(boundaries[1:-1], Y_rot)
    - Dequantize: Y_hat = centroids[indices]
    - Inverse rotate & rescale: X_recon = (Y_hat @ Pi) * X.norms
    - Compute D_prod, RMSE, correlation
    """
    # Normalize X to unit vectors
    X_norms = X.norm(dim=1, keepdim=True)
    X_unit = X / (X_norms + 1e-10)
    
    # Rotate
    Y_rot = X_unit @ Pi.T
    
    # Quantize
    interior_boundaries = boundaries[1:-1]
    indices = torch.searchsorted(interior_boundaries, Y_rot)
    
    # Dequantize
    Y_hat = centroids[indices]
    
    # Inverse rotate & rescale
    X_recon = (Y_hat @ Pi) * X_norms
    
    # Compute true inner products: ip_true = (X * Y).sum(dim=1)
    ip_true = (X * Y).sum(dim=1)
    
    # Compute quantized inner products: ip_quant = (X_recon * Y).sum(dim=1)
    ip_quant = (X_recon * Y).sum(dim=1)
    
    # Compute D_prod: ((ip_true - ip_quant)**2).mean().item()
    d_prod = ((ip_true - ip_quant) ** 2).mean().item()
    
    # Relative RMSE
    true_ip_var = (ip_true ** 2).mean().item()
    relative_rmse = np.sqrt(d_prod / max(true_ip_var, 1e-10))
    
    # Pearson correlation
    correlation = torch.corrcoef(torch.stack([ip_true, ip_quant]))[0, 1].item()
    
    return {
        "d_prod": d_prod,
        "relative_rmse": relative_rmse,
        "correlation": correlation,
    }


def compute_variance_ratio(X: torch.Tensor, Pi: torch.Tensor) -> float:
    """Compute post-rotation variance ratio (max/min per-coordinate variance)."""
    X_unit = X / (X.norm(dim=1, keepdim=True) + 1e-10)
    Y_rot = X_unit @ Pi.T
    variances = Y_rot.var(dim=0)
    ratio = (variances.max() / variances.clamp(min=1e-10).min()).item()
    return ratio


def main():
    print("=" * 60)
    print("Experiment 2 Fixed: Manual D_prod Comparison")
    print("Dense vs Hadamard Rotations (CPU-only)")
    print("=" * 60)
    
    # Data Generation
    print("\nGenerating anisotropic data...")
    n_samples = 5000
    d = 128
    
    # Generate anisotropic structure: variance ratio ~3-5x across channels
    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)
    
    # Create variance profile with ~4x ratio
    base_variance = torch.linspace(0.5, 2.0, d, device=device)
    variance_profile = base_variance / base_variance.mean()  # Normalize to mean=1
    
    # Generate vectors with anisotropic variance
    X_aniso = torch.randn(n_samples, d, device=device, dtype=torch.float32, generator=rng)
    X_aniso = X_aniso * variance_profile.sqrt()
    
    # Normalize all vectors to unit sphere
    X_all = X_aniso / X_aniso.norm(dim=1, keepdim=True)
    
    # Split into two halves: X (keys to quantize) and Y (queries kept exact)
    X = X_all[:n_samples//2]
    Y = X_all[n_samples//2:n_samples]
    
    print(f"  Generated {X.shape[0]} key vectors and {Y.shape[0]} query vectors")
    print(f"  Dimension: {d}, Anisotropy ratio: ~{variance_profile.max()/variance_profile.min():.1f}x")
    
    # Load codebook
    print("\nLoading 3-bit codebook...")
    codebook_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  "turboquant/codebooks/codebook_d128_b3.json")
    with open(codebook_path, 'r') as f:
        codebook_data = json.load(f)
    
    boundaries = torch.tensor(codebook_data["boundaries"], device=device, dtype=torch.float32)
    centroids = torch.tensor(codebook_data["centroids"], device=device, dtype=torch.float32)
    print(f"  Loaded codebook: {len(centroids)} centroids, {len(boundaries)} boundaries")
    
    # Generate rotation matrices
    print("\nGenerating rotation matrices...")
    Pi_dense = dense_rotation(d, device)
    Pi_hadamard = hadamard_rotation(d, device)
    
    # Assert orthogonality
    print("\nOrthogonality checks:")
    assert_orthogonality(Pi_dense, "Dense")
    assert_orthogonality(Pi_hadamard, "Hadamard")
    
    # Compute variance ratios
    print("\nPost-rotation variance analysis:")
    var_ratio_dense = compute_variance_ratio(X, Pi_dense)
    var_ratio_hadamard = compute_variance_ratio(X, Pi_hadamard)
    print(f"  Dense rotation variance ratio: {var_ratio_dense:.4f}")
    print(f"  Hadamard rotation variance ratio: {var_ratio_hadamard:.4f}")
    
    # Compute D_prod for both rotations on the SAME X and Y tensors
    print("\nComputing D_prod metrics...")
    results_dense = compute_d_prod_manual(X, Y, Pi_dense, boundaries, centroids)
    results_hadamard = compute_d_prod_manual(X, Y, Pi_hadamard, boundaries, centroids)
    
    # Calculate improvement
    d_prod_dense = results_dense["d_prod"]
    d_prod_hadamard = results_hadamard["d_prod"]
    improvement_pct = ((d_prod_dense - d_prod_hadamard) / d_prod_dense) * 100
    
    # Print terminal summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Dense':>15} {'Hadamard':>15}")
    print("-" * 60)
    print(f"{'Orthogonality Error':<25} {torch.max(torch.abs(Pi_dense @ Pi_dense.T - torch.eye(d, device=device))):>15.6f} {torch.max(torch.abs(Pi_hadamard @ Pi_hadamard.T - torch.eye(d, device=device))):>15.6f}")
    print(f"{'Variance Ratio':<25} {var_ratio_dense:>15.4f} {var_ratio_hadamard:>15.4f}")
    print(f"{'D_prod':<25} {d_prod_dense:>15.6f} {d_prod_hadamard:>15.6f}")
    print(f"{'Relative RMSE':<25} {results_dense['relative_rmse']:>15.4f} {results_hadamard['relative_rmse']:>15.4f}")
    print(f"{'Pearson Correlation':<25} {results_dense['correlation']:>15.4f} {results_hadamard['correlation']:>15.4f}")
    print("-" * 60)
    print(f"{'% Improvement':<25} {improvement_pct:>15.2f}%")
    
    # Save results to JSON
    output_results = {
        "dense": {
            "orthogonality_error": torch.max(torch.abs(Pi_dense @ Pi_dense.T - torch.eye(d, device=device))).item(),
            "variance_ratio": var_ratio_dense,
            "d_prod": d_prod_dense,
            "relative_rmse": results_dense["relative_rmse"],
            "correlation": results_dense["correlation"],
        },
        "hadamard": {
            "orthogonality_error": torch.max(torch.abs(Pi_hadamard @ Pi_hadamard.T - torch.eye(d, device=device))).item(),
            "variance_ratio": var_ratio_hadamard,
            "d_prod": d_prod_hadamard,
            "relative_rmse": results_hadamard["relative_rmse"],
            "correlation": results_hadamard["correlation"],
        },
        "improvement_pct": improvement_pct,
        "config": {
            "n_samples": n_samples,
            "dimension": d,
            "bits": 3,
            "seed": 42,
        }
    }
    
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 "experiment2_fixed_results.json")
    with open(results_path, 'w') as f:
        json.dump(output_results, f, indent=2, sort_keys=True)
    print(f"\nResults saved to: {results_path}")
    
    # Final status line
    if improvement_pct > 0:
        print(f"\nSUCCESS: Hadamard improves D_prod by {improvement_pct:.2f}%")
    else:
        print(f"\nFAILED: Check implementation (Hadamard is {-improvement_pct:.2f}% worse)")


if __name__ == "__main__":
    main()
