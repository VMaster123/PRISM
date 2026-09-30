"""
PRISM 2D Heat Conduction PDE Solver & SIMP Topology Optimization Engine
Governing PDE: -\nabla \cdot (\kappa(\rho) \nabla u) = f in \Omega
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Tuple, Dict, List, Optional


class HeatPDE2D:
    """
    2D Steady-State Heat Conduction Finite Difference PDE Solver.
    
    Domain: Unit square [0, 1] x [0, 1] discretized on an Nx x Ny grid.
    PDE: -\nabla \cdot (\kappa(\rho) \nabla u) = f
    
    Material Interpolation (SIMP):
    \kappa(\rho) = \kappa_{min} + (\kappa_{max} - \kappa_{min}) * \rho^p
    """
    def __init__(
        self,
        nx: int = 32,
        ny: int = 32,
        k_min: float = 1e-3,
        k_max: float = 1.0,
        simp_p: float = 3.0,
        sink_location: str = "bottom_center",  # 'bottom_center', 'center', 'left_edge'
        heat_source_type: str = "uniform"       # 'uniform', 'center_spot', 'distributed'
    ):
        self.nx = nx
        self.ny = ny
        self.n_nodes = nx * ny
        self.k_min = k_min
        self.k_max = k_max
        self.simp_p = simp_p
        self.sink_location = sink_location
        self.heat_source_type = heat_source_type
        
        # Grid spacing (assume Lx = Ly = 1.0)
        self.dx = 1.0 / (nx - 1)
        self.dy = 1.0 / (ny - 1)
        
        # Dirichlet boundary conditions (Heat Sinks u = 0)
        self.dirichlet_mask = np.zeros((ny, nx), dtype=bool)
        self._setup_boundary_conditions()
        
        # Volumetric Heat Source f
        self.f_field = self._setup_heat_source()
        
    def _setup_boundary_conditions(self):
        """Define Dirichlet sink nodes (u = 0)."""
        ny, nx = self.ny, self.nx
        if self.sink_location == "bottom_center":
            # Central 20% of bottom edge
            start_x = int(0.4 * nx)
            end_x = int(0.6 * nx)
            self.dirichlet_mask[0, start_x:end_x] = True
        elif self.sink_location == "center":
            # Central 10% patch
            cx, cy = nx // 2, ny // 2
            w = max(1, nx // 8)
            self.dirichlet_mask[cy-w:cy+w, cx-w:cx+w] = True
        elif self.sink_location == "left_edge":
            # Entire left boundary
            self.dirichlet_mask[:, 0] = True
        else:
            # Default to bottom center
            self.dirichlet_mask[0, nx//4:3*nx//4] = True

    def _setup_heat_source(self) -> np.ndarray:
        """Define volumetric heat source f(x, y)."""
        f = np.ones((self.ny, self.nx), dtype=np.float64)
        if self.heat_source_type == "center_spot":
            cy, cx = self.ny // 2, self.nx // 2
            y, x = np.ogrid[:self.ny, :self.nx]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            f = np.exp(-dist**2 / (0.1 * self.nx)**2)
        elif self.heat_source_type == "distributed":
            y, x = np.ogrid[:self.ny, :self.nx]
            f = 1.0 + 0.5 * np.sin(2 * np.pi * x / self.nx) * np.cos(2 * np.pi * y / self.ny)
        return f

    def compute_conductivity(self, rho: np.ndarray) -> np.ndarray:
        """Compute SIMP material thermal conductivity map."""
        rho_clamped = np.clip(rho, 0.0, 1.0)
        return self.k_min + (self.k_max - self.k_min) * (rho_clamped ** self.simp_p)

    def assemble_stiffness_matrix(self, k_map: np.ndarray) -> sp.csr_matrix:
        """
        Assemble sparse conductivity matrix K for finite difference 5-point stencil:
        -d/dx (k du/dx) - d/dy (k du/dy) = f
        """
        ny, nx = self.ny, self.nx
        dx, dy = self.dx, self.dy

        row_ind, col_ind, data = [], [], []

        for j in range(ny):
            for i in range(nx):
                node = j * nx + i

                if self.dirichlet_mask[j, i]:
                    # Dirichlet boundary: u = 0 -> equation: 1.0 * u_{node} = 0
                    row_ind.append(node)
                    col_ind.append(node)
                    data.append(1.0)
                    continue

                diag_val = 0.0

                # Left neighbor (i - 1, j)
                if i > 0:
                    k_edge = 2.0 * k_map[j, i] * k_map[j, i - 1] / (k_map[j, i] + k_map[j, i - 1] + 1e-12)
                    coeff = k_edge / (dx ** 2)
                    diag_val += coeff
                    if not self.dirichlet_mask[j, i - 1]:
                        row_ind.append(node)
                        col_ind.append(node - 1)
                        data.append(-coeff)
                    else:
                        # Dirichlet neighbor contributes 0 to RHS since u_dirichlet = 0
                        pass

                # Right neighbor (i + 1, j)
                if i < nx - 1:
                    k_edge = 2.0 * k_map[j, i] * k_map[j, i + 1] / (k_map[j, i] + k_map[j, i + 1] + 1e-12)
                    coeff = k_edge / (dx ** 2)
                    diag_val += coeff
                    if not self.dirichlet_mask[j, i + 1]:
                        row_ind.append(node)
                        col_ind.append(node + 1)
                        data.append(-coeff)

                # Bottom neighbor (i, j - 1)
                if j > 0:
                    k_edge = 2.0 * k_map[j, i] * k_map[j - 1, i] / (k_map[j, i] + k_map[j - 1, i] + 1e-12)
                    coeff = k_edge / (dy ** 2)
                    diag_val += coeff
                    if not self.dirichlet_mask[j - 1, i]:
                        row_ind.append(node)
                        col_ind.append(node - nx)
                        data.append(-coeff)

                # Top neighbor (i, j + 1)
                if j < ny - 1:
                    k_edge = 2.0 * k_map[j, i] * k_map[j + 1, i] / (k_map[j, i] + k_map[j + 1, i] + 1e-12)
                    coeff = k_edge / (dy ** 2)
                    diag_val += coeff
                    if not self.dirichlet_mask[j + 1, i]:
                        row_ind.append(node)
                        col_ind.append(node + nx)
                        data.append(-coeff)

                # Diagonal term
                row_ind.append(node)
                col_ind.append(node)
                data.append(diag_val)

        K = sp.csr_matrix((data, (row_ind, col_ind)), shape=(self.n_nodes, self.n_nodes))
        return K

    def solve(self, rho: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Solve forward PDE for state u and compute thermal compliance performance J = f^T u.
        Returns:
            u_field: (ny, nx) temperature grid
            J: thermal compliance (lower is better, higher thermal efficiency)
        """
        k_map = self.compute_conductivity(rho)
        K = self.assemble_stiffness_matrix(k_map)
        
        # Build RHS vector f
        rhs = self.f_field.ravel().copy()
        # Enforce Dirichlet u = 0 on sink nodes
        rhs[self.dirichlet_mask.ravel()] = 0.0
        
        # Solve K * u = rhs
        u_vec = spla.spsolve(K, rhs)
        u_field = u_vec.reshape((self.ny, self.nx))
        
        # Compliance performance J = integral(f * u) approx sum(f * u * dx * dy)
        J = float(np.sum(self.f_field * u_field) * self.dx * self.dy)
        return u_field, J


class SIMPOptimizer:
    """
    Topology Optimization via SIMP (Solid Isotropic Material with Penalization)
    and Optimality Criteria (OC) update for Heat Dissipation Design.
    """
    def __init__(
        self,
        pde_solver: HeatPDE2D,
        vol_frac: float = 0.3,
        max_iter: int = 40,
        move_limit: float = 0.2,
        damping: float = 0.5
    ):
        self.pde = pde_solver
        self.vol_frac = vol_frac
        self.max_iter = max_iter
        self.move = move_limit
        self.eta = damping

    def optimize(self, initial_rho: Optional[np.ndarray] = None) -> Dict:
        """
        Run SIMP optimization loop.
        Returns dictionary of history and final optimized design rho.
        """
        ny, nx = self.pde.ny, self.pde.nx
        if initial_rho is None:
            # Uniform initial density equal to target volume fraction + minor noise
            rho = np.full((ny, nx), self.vol_frac)
            # Add slight perturbation to break symmetry
            rho += 0.02 * (np.random.rand(ny, nx) - 0.5)
            rho = np.clip(rho, 0.0, 1.0)
        else:
            rho = initial_rho.copy()

        history_J = []
        history_rho = [rho.copy()]

        for it in range(self.max_iter):
            # Solve PDE
            u_field, J = self.pde.solve(rho)
            history_J.append(J)

            # Compute sensitivities dJ/drho
            # For heat compliance: J = f^T u. Derivative w.r.t. element e conductivity:
            # dJ/drho_e = - p * (k_max - k_min) * rho_e^(p-1) * (|grad u|^2)
            grad_u_y, grad_u_x = np.gradient(u_field, self.pde.dy, self.pde.dx)
            grad_u_sq = grad_u_x**2 + grad_u_y**2
            
            p = self.pde.simp_p
            dk_drho = p * (self.pde.k_max - self.pde.k_min) * (np.maximum(rho, 1e-4) ** (p - 1))
            sensitivity = dk_drho * grad_u_sq
            sensitivity = np.maximum(sensitivity, 1e-10)

            # Optimality Criteria (OC) Update step with bisection search for Lagrange multiplier lambda
            l1, l2 = 1e-10, 1e10
            rho_new = rho.copy()

            while (l2 - l1) / (l1 + l2) > 1e-4:
                l_mid = 0.5 * (l1 + l2)
                # OC update formula: rho_new = rho * (sensitivity / lambda)^eta
                B = (sensitivity / l_mid) ** self.eta
                rho_candidate = np.clip(
                    rho * B,
                    np.maximum(0.0, rho - self.move),
                    np.minimum(1.0, rho + self.move)
                )
                if np.mean(rho_candidate) - self.vol_frac > 0:
                    l1 = l_mid
                else:
                    l2 = l_mid
                rho_new = rho_candidate

            rho = rho_new
            history_rho.append(rho.copy())

        return {
            "final_rho": rho,
            "final_u": u_field,
            "final_J": J,
            "history_J": history_J,
            "history_rho": history_rho
        }


def generate_pde_dataset(
    num_samples: int = 150,
    nx: int = 32,
    ny: int = 32,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Generate a diverse dataset of optimized and partially-optimized PDE design fields.
    Varies boundary conditions, volume fractions, heat source distributions, and optimization stages.
    """
    np.random.seed(seed)
    designs = []
    compliance_list = []
    metadata = []

    sink_configs = ["bottom_center", "center", "left_edge"]
    source_configs = ["uniform", "center_spot", "distributed"]
    vol_fractions = [0.20, 0.30, 0.40, 0.50]

    count = 0
    while count < num_samples:
        sink = str(np.random.choice(sink_configs))
        source = str(np.random.choice(source_configs))
        vol = float(np.random.choice(vol_fractions))
        max_iter = int(np.random.randint(15, 45))  # include early-stopped & converged

        solver = HeatPDE2D(nx=nx, ny=ny, sink_location=sink, heat_source_type=source)
        optimizer = SIMPOptimizer(solver, vol_frac=vol, max_iter=max_iter)

        res = optimizer.optimize()
        
        # Save both final state and a couple intermediate topology states for high library diversity
        hist_rho = res["history_rho"]
        hist_J = res["history_J"]
        
        # Sample final design
        designs.append(hist_rho[-1])
        compliance_list.append(hist_J[-1])
        metadata.append({"sink": sink, "source": source, "vol": vol, "iter": max_iter})
        count += 1

        if count < num_samples and len(hist_rho) > 10:
            # Sample an intermediate early-terminated design (HiLAB style)
            mid_idx = len(hist_rho) // 2
            designs.append(hist_rho[mid_idx])
            compliance_list.append(hist_J[mid_idx])
            metadata.append({"sink": sink, "source": source, "vol": vol, "iter": mid_idx})
            count += 1

    designs_arr = np.array(designs[:num_samples], dtype=np.float32)  # Shape (N, H, W)
    compliance_arr = np.array(compliance_list[:num_samples], dtype=np.float32)

    return {
        "designs": designs_arr,             # Shape: (N, 32, 32)
        "compliance": compliance_arr,       # Shape: (N,)
        "metadata": metadata[:num_samples]
    }
