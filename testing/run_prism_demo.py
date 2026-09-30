"""
PRISM: Physics-Representation Interpretability for Structural Mechanics & PDE Design
End-to-End Execution Script:
1. PDE Simulation & SIMP Topology Optimization Dataset Generation
2. Base Spatial Autoencoder Training (rho -> z -> \hat{rho})
3. Sparse Autoencoder Training (SAE with L1 penalty per arXiv:2309.08600)
4. Feature Extraction & Monosemanticity Analysis
5. Controlled Interventions & Causal Physics Testing (f_k -> \tilde{\rho} -> \tilde{u} -> \tilde{J})
6. Publication-Quality Visualizations
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Import PRISM package components
from prism.pde_solver import HeatPDE2D, SIMPOptimizer, generate_pde_dataset
from prism.models import SpatialAutoencoder, SparseAutoencoder
from prism.interpretability import PRISMInterventionEngine


def main():
    print("==========================================================================")
    print("  PRISM: Physics-Representation Interpretability for PDE-Constrained Design")
    print("==========================================================================")

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using compute device: {device}")

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Generate PDE Optimization Dataset
    # -------------------------------------------------------------------------
    num_samples = 120
    grid_size = 32
    print(f"\n[STEP 1] Generating {num_samples} topology-optimized PDE design fields ({grid_size}x{grid_size})...")
    start_time = time.time()
    
    pde_data = generate_pde_dataset(num_samples=num_samples, nx=grid_size, ny=grid_size, seed=42)
    designs = pde_data["designs"]  # Shape: (N, 32, 32)
    compliance = pde_data["compliance"]
    
    print(f" -> Dataset generated in {time.time() - start_time:.2f}s")
    print(f" -> Designs shape: {designs.shape}, Density range: [{designs.min():.3f}, {designs.max():.3f}]")
    print(f" -> Thermal Compliance J range: [{compliance.min():.4f}, {compliance.max():.4f}]")

    # Plot sample designs from dataset
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("PRISM Dataset: PDE Topology-Optimized Heat Dissipation Structures (\\rho)", fontsize=14, fontweight='bold')
    for i, ax in enumerate(axes.flat):
        if i < len(designs):
            im = ax.imshow(designs[i], cmap="magma", vmin=0, vmax=1)
            ax.set_title(f"Design #{i}\nJ = {compliance[i]:.3f}", fontsize=10)
            ax.axis("off")
    plt.colorbar(im, ax=axes.ravel().tolist(), label="Material Density \\rho")
    dataset_fig_path = os.path.join(results_dir, "fig1_pde_dataset_samples.png")
    plt.savefig(dataset_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" -> Saved dataset visualization to {dataset_fig_path}")

    # Prepare PyTorch DataLoader
    designs_tensor = torch.from_numpy(designs).unsqueeze(1).float()  # (N, 1, 32, 32)
    dataset = TensorDataset(designs_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # -------------------------------------------------------------------------
    # Step 2: Train Base Spatial Autoencoder (\phi: \rho -> z -> \hat{\rho})
    # -------------------------------------------------------------------------
    latent_dim = 16
    print(f"\n[STEP 2] Training Base Spatial Autoencoder (Latent dim = {latent_dim})...")
    base_ae = SpatialAutoencoder(in_channels=1, latent_dim=latent_dim, img_size=grid_size).to(device)
    ae_optimizer = optim.AdamW(base_ae.parameters(), lr=1e-3, weight_decay=1e-4)
    ae_criterion = nn.MSELoss()

    ae_epochs = 60
    base_ae.train()
    for epoch in range(1, ae_epochs + 1):
        total_loss = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            x_hat, z = base_ae(batch_x)
            loss = ae_criterion(x_hat, batch_x)
            
            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        mean_loss = total_loss / len(dataset)
        if epoch % 15 == 0 or epoch == ae_epochs:
            print(f"  Epoch {epoch:2d}/{ae_epochs} | Base AE Recon MSE: {mean_loss:.6f}")

    # Extract all latent vectors z
    base_ae.eval()
    with torch.no_grad():
        all_z = base_ae.encode(designs_tensor.to(device))  # (N, latent_dim)
    print(f" -> Encoded dataset to latent vectors z of shape {all_z.shape}")

    # -------------------------------------------------------------------------
    # Step 3: Train Sparse Autoencoder (SAE with L1 penalty per arXiv:2309.08600)
    # -------------------------------------------------------------------------
    num_features = 64  # Overcomplete dictionary size K = 64 > d = 16
    l1_coeff = 0.015
    print(f"\n[STEP 3] Training Sparse Autoencoder (SAE: {latent_dim} -> {num_features} -> {latent_dim}, L1 coeff = {l1_coeff})...")
    
    sae = SparseAutoencoder(latent_dim=latent_dim, num_features=num_features, l1_coeff=l1_coeff).to(device)
    sae_optimizer = optim.Adam(sae.parameters(), lr=2e-3)
    
    z_dataset = TensorDataset(all_z)
    z_loader = DataLoader(z_dataset, batch_size=16, shuffle=True)

    sae_epochs = 80
    sae_history = {"total": [], "recon": [], "l1": [], "l0": []}
    
    sae.train()
    for epoch in range(1, sae_epochs + 1):
        ep_recon, ep_l1, ep_l0, ep_total = 0.0, 0.0, 0.0, 0.0
        for (batch_z,) in z_loader:
            z_hat, f, loss_dict = sae(batch_z)
            
            sae_optimizer.zero_grad()
            loss_dict["total_loss"].backward()
            sae_optimizer.step()
            
            # Project decoder columns to unit L2 norm as required by SAE dictionary learning
            sae.normalize_decoder_weights()
            
            bs = batch_z.size(0)
            ep_total += loss_dict["total_loss"].item() * bs
            ep_recon += loss_dict["recon_loss"].item() * bs
            ep_l1 += loss_dict["l1_loss"].item() * bs
            ep_l0 += loss_dict["l0_sparsity"].item() * bs

        n = len(z_dataset)
        sae_history["total"].append(ep_total / n)
        sae_history["recon"].append(ep_recon / n)
        sae_history["l1"].append(ep_l1 / n)
        sae_history["l0"].append(ep_l0 / n)

        if epoch % 20 == 0 or epoch == sae_epochs:
            print(f"  Epoch {epoch:2d}/{sae_epochs} | SAE Total: {ep_total/n:.5f} | Recon MSE: {ep_recon/n:.5f} | L0 Sparsity: {ep_l0/n:.2f}/{num_features} active features")

    # Plot SAE Training & Sparsity Convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(sae_history["recon"], label="SAE Recon MSE", color="crimson", linewidth=2)
    ax1.plot(sae_history["total"], label="Total SAE Loss", color="navy", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("SAE Dictionary Learning Convergence")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(sae_history["l0"], label="L0 Active Features", color="forestgreen", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Avg Active Sparse Features per Design")
    ax2.set_title("Sparse Representation Feature Activation (L0 Norm)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    sae_fig_path = os.path.join(results_dir, "fig2_sae_training_convergence.png")
    plt.savefig(sae_fig_path, dpi=300)
    plt.close()
    print(f" -> Saved SAE convergence visualization to {sae_fig_path}")

    # -------------------------------------------------------------------------
    # Step 4: Feature Extraction & Monosemanticity Analysis
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Extracting Sparse Features & Performing Monosemanticity Analysis...")
    engine = PRISMInterventionEngine(base_ae=base_ae, sae=sae, device=str(device))
    top_feats = engine.analyze_top_features(designs, top_k_features=4)

    print(" -> Top Disentangled Geometric Features Discovered by SAE:")
    for f_idx, info in top_feats.items():
        print(f"    Feature #{f_idx:2d} | Max Activation: {info['max_act']:.3f} | Top Sample IDs: {info['top_sample_indices'][:3]}")

    # -------------------------------------------------------------------------
    # Step 5: Controlled Interventions & Causal Physics Testing
    # f_k -> \tilde{z} -> \tilde{\rho} -> \tilde{u} -> \tilde{J}
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Running PRISM Controlled Interventions (Geometry <-> Representation <-> Physics)...")
    
    # Pick a baseline design (sample #0)
    test_idx = 0
    rho_orig = designs[test_idx]
    
    # Pick one of the top active SAE features
    target_feat_idx = list(top_feats.keys())[0]
    
    # Run boosting intervention (amplifying feature f_k)
    res_boost = engine.intervene_and_simulate(
        rho_orig=rho_orig,
        feature_idx=target_feat_idx,
        alpha=2.5,
        mode="boost"
    )

    # Run ablation intervention (suppressing feature f_k to 0)
    res_ablate = engine.intervene_and_simulate(
        rho_orig=rho_orig,
        feature_idx=target_feat_idx,
        mode="ablate"
    )

    print(f"\n --- INTERVENTION RESULTS FOR FEATURE #{target_feat_idx} ---")
    print(f" Baseline Design #{test_idx} | Thermal Compliance J_orig = {res_boost['J_orig']:.4f}")
    print(f" 1. BOOST Feature #{target_feat_idx} (Val: {res_boost['orig_feat_val']:.3f} -> {res_boost['mod_feat_val']:.3f}):")
    print(f"    -> Modified J_mod = {res_boost['J_mod']:.4f} | Delta J = {res_boost['delta_J']:+.4f} ({res_boost['rel_change_J_pct']:+.2f}%)")
    print(f" 2. ABLATE Feature #{target_feat_idx} (Val: {res_ablate['orig_feat_val']:.3f} -> 0.0):")
    print(f"    -> Modified J_mod = {res_ablate['J_mod']:.4f} | Delta J = {res_ablate['delta_J']:+.4f} ({res_ablate['rel_change_J_pct']:+.2f}%)")

    # -------------------------------------------------------------------------
    # Step 6: Comprehensive PRISM Triad Visualization
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"PRISM Interpretability Triad: Feature #{target_feat_idx} Intervention Test\n"
        f"[ Geometry (\\rho)  <--->  Representation (f_{target_feat_idx})  <--->  Physics (u, J) ]",
        fontsize=14, fontweight="bold"
    )

    # Row 1: Baseline Physics & Geometry
    ax1 = fig.add_subplot(2, 4, 1)
    im1 = ax1.imshow(res_boost["rho_orig"], cmap="magma", vmin=0, vmax=1)
    ax1.set_title(f"Baseline Geometry \\rho_{{orig}}\n(Design #{test_idx})")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 4, 2)
    im2 = ax2.imshow(res_boost["u_orig"], cmap="inferno")
    ax2.set_title(f"Baseline Temperature u_{{orig}}\nJ_{{orig}} = {res_boost['J_orig']:.4f}")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Row 1: Feature Boosted Geometry & Physics
    ax3 = fig.add_subplot(2, 4, 3)
    im3 = ax3.imshow(res_boost["rho_mod"], cmap="magma", vmin=0, vmax=1)
    ax3.set_title(f"Boosted Geometry \\tilde{{\\rho}}\n(f_{{{target_feat_idx}}} += 2.5)")
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(2, 4, 4)
    im4 = ax4.imshow(res_boost["u_mod"], cmap="inferno")
    ax4.set_title(f"Boosted Temperature \\tilde{{u}}\n\\tilde{{J}} = {res_boost['J_mod']:.4f} ({res_boost['rel_change_J_pct']:+.1f}%)")
    ax4.axis("off")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # Row 2: Geometric & Physical Impact Maps (\Delta \rho and \Delta u)
    ax5 = fig.add_subplot(2, 4, 5)
    im5 = ax5.imshow(res_boost["delta_rho"], cmap="bwr", vmin=-0.5, vmax=0.5)
    ax5.set_title(f"Geometric Shift \\Delta\\rho\n(\\tilde{{\\rho}} - \\rho_{{orig}})")
    ax5.axis("off")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(2, 4, 6)
    im6 = ax6.imshow(res_boost["delta_u"], cmap="coolwarm")
    ax6.set_title(f"Thermal Response Shift \\Delta u\n(\\tilde{{u}} - u_{{orig}})")
    ax6.axis("off")
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    # Row 2: Feature Ablated Geometry & Physics
    ax7 = fig.add_subplot(2, 4, 7)
    im7 = ax7.imshow(res_ablate["rho_mod"], cmap="magma", vmin=0, vmax=1)
    ax7.set_title(f"Ablated Geometry \\tilde{{\\rho}}\n(f_{{{target_feat_idx}}} = 0)")
    ax7.axis("off")
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

    ax8 = fig.add_subplot(2, 4, 8)
    im8 = ax8.imshow(res_ablate["u_mod"], cmap="inferno")
    ax8.set_title(f"Ablated Temperature \\tilde{{u}}\n\\tilde{{J}} = {res_ablate['J_mod']:.4f} ({res_ablate['rel_change_J_pct']:+.1f}%)")
    ax8.axis("off")
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

    plt.tight_layout()
    triad_fig_path = os.path.join(results_dir, "fig3_prism_triad_intervention.png")
    plt.savefig(triad_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" -> Saved PRISM Triad Intervention visualization to {triad_fig_path}")

    print("\n==========================================================================")
    print("  PRISM Starter Code & Interpretability Execution Completed Successfully!")
    print(f"  All results saved in '{results_dir}/'")
    print("==========================================================================")


if __name__ == "__main__":
    main()
