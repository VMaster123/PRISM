"""
PRISM Interpretability & Controlled Intervention Engine
Realizes the Triad: Geometry (rho) <---> Representation (z, f_k) <---> Physics (u, J)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from .pde_solver import HeatPDE2D
from .models import SpatialAutoencoder, SparseAutoencoder


class PRISMInterventionEngine:
    """
    Engine for extracting interpretable sparse features and performing 
    controlled interventions on PDE-constrained engineering designs.
    """
    def __init__(
        self,
        base_ae: SpatialAutoencoder,
        sae: SparseAutoencoder,
        pde_solver: Optional[HeatPDE2D] = None,
        device: str = "cpu"
    ):
        self.base_ae = base_ae.to(device).eval()
        self.sae = sae.to(device).eval()
        self.pde_solver = pde_solver or HeatPDE2D()
        self.device = device

    @torch.no_grad()
    def get_latent_and_features(self, designs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode spatial design density maps rho into base latent z and SAE sparse features f.
        Input: designs of shape (N, H, W) or (N, 1, H, W)
        Returns:
            z_all: (N, latent_dim)
            f_all: (N, num_features)
        """
        if designs.ndim == 3:
            designs_tensor = torch.from_numpy(designs).unsqueeze(1).float().to(self.device)
        else:
            designs_tensor = torch.from_numpy(designs).float().to(self.device)

        z = self.base_ae.encode(designs_tensor)
        f = self.sae.encode(z)

        return z.cpu().numpy(), f.cpu().numpy()

    def analyze_top_features(self, designs: np.ndarray, top_k_features: int = 5) -> Dict:
        """
        Identify the most active and variable SAE features across the dataset.
        Returns top feature indices and their top-activating design indices.
        """
        _, f_all = self.get_latent_and_features(designs)
        mean_activation = np.mean(f_all, axis=0)
        var_activation = np.var(f_all, axis=0)
        max_activation = np.max(f_all, axis=0)

        # Rank features by variance * max activation to find meaningful features
        importance_score = var_activation * max_activation
        top_feature_indices = np.argsort(importance_score)[::-1][:top_k_features]

        feature_analysis = {}
        for feat_idx in top_feature_indices:
            activations = f_all[:, feat_idx]
            top_sample_indices = np.argsort(activations)[::-1][:5]
            feature_analysis[int(feat_idx)] = {
                "max_act": float(max_activation[feat_idx]),
                "mean_act": float(mean_activation[feat_idx]),
                "var_act": float(var_activation[feat_idx]),
                "top_sample_indices": top_sample_indices.tolist(),
                "top_activations": activations[top_sample_indices].tolist()
            }

        return feature_analysis

    @torch.no_grad()
    def intervene_and_simulate(
        self,
        rho_orig: np.ndarray,
        feature_idx: int,
        alpha: float = 2.0,
        mode: str = "boost",  # 'boost', 'ablate', or 'set_value'
        pde_solver: Optional[HeatPDE2D] = None
    ) -> Dict:
        """
        Perform a controlled intervention on sparse feature f_k and evaluate physical consequence:
        f_k -> \tilde{z} -> \tilde{\rho} -> \tilde{u} -> \tilde{J}
        
        Args:
            rho_orig: Original spatial design field (H, W)
            feature_idx: Index k of the SAE feature to intervene on
            alpha: Magnitude of intervention (multiplier for boost, or value for set_value)
            mode: 'boost' (f_k += alpha), 'ablate' (f_k = 0), or 'set_value' (f_k = alpha)
            pde_solver: PDE solver instance (uses default if None)
            
        Returns:
            Dict containing original and modified geometry, physical state, and performance J.
        """
        solver = pde_solver or self.pde_solver
        
        # 1. Evaluate baseline physical state
        u_orig, J_orig = solver.solve(rho_orig)

        # 2. Encode baseline design to z and f
        rho_tensor = torch.from_numpy(rho_orig).unsqueeze(0).unsqueeze(0).float().to(self.device)
        z_orig = self.base_ae.encode(rho_tensor)
        f_orig = self.sae.encode(z_orig)

        # 3. Apply Intervention I_k(f)
        f_mod = f_orig.clone()
        orig_val = float(f_orig[0, feature_idx].item())

        if mode == "ablate":
            f_mod[0, feature_idx] = 0.0
            new_val = 0.0
        elif mode == "boost":
            f_mod[0, feature_idx] = f_orig[0, feature_idx] + alpha
            new_val = float(f_mod[0, feature_idx].item())
        elif mode == "set_value":
            f_mod[0, feature_idx] = alpha
            new_val = float(alpha)
        else:
            raise ValueError(f"Unknown intervention mode: {mode}")

        # 4. Decode modified feature vector back to latent space \tilde{z} and spatial geometry \tilde{\rho}
        z_mod = self.sae.decode(f_mod)
        rho_mod_tensor = self.base_ae.decode(z_mod)
        rho_mod = rho_mod_tensor.squeeze().cpu().numpy()
        rho_mod = np.clip(rho_mod, 0.0, 1.0)

        # 5. Solve PDE on modified geometry \tilde{\rho} to get \tilde{u} and \tilde{J}
        u_mod, J_mod = solver.solve(rho_mod)

        # 6. Compute deltas
        delta_rho = rho_mod - rho_orig
        delta_u = u_mod - u_orig
        delta_J = J_mod - J_orig
        rel_change_J = (J_mod - J_orig) / (J_orig + 1e-12) * 100.0

        return {
            "feature_idx": feature_idx,
            "mode": mode,
            "orig_feat_val": orig_val,
            "mod_feat_val": new_val,
            "rho_orig": rho_orig,
            "u_orig": u_orig,
            "J_orig": J_orig,
            "rho_mod": rho_mod,
            "u_mod": u_mod,
            "J_mod": J_mod,
            "delta_rho": delta_rho,
            "delta_u": delta_u,
            "delta_J": delta_J,
            "rel_change_J_pct": rel_change_J
        }
