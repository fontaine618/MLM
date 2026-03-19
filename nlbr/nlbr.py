import torch
from torch import Tensor
from torch.nn import Parameter


class NLR(torch.nn.Module):
    """Nonlinear Regression (non-Bayesian).

    Fits  U = a + b(x) + e  in kernel PCA feature space using ordinary
    least squares:

        a  = mean of the training responses  (per-row intercept)
        b  = KPCA regression with no prior and no regularisation beyond
             spectral truncation at ``variance_explained``

    This is the non-Bayesian counterpart of NLBR: there are no priors,
    no variance ratios, and no ``power`` parameter.
    """

    def __init__(
            self,
            Kx_train_train: Tensor,
            variance_explained: float = 0.99,
    ):
        """Initialise NLR and precompute its spectral state.

        Args:
            Kx_train_train: Training kernel matrix with shape ``(N, N)``.
            variance_explained: Fraction of cumulative variance to retain in
                the eigenspectrum (in ``[0, 1]``).

        Raises:
            ValueError: If ``variance_explained`` is outside ``[0, 1]``.
        """
        super().__init__()
        self.register_buffer("_evals", None)
        self.register_buffer("_evecs", None)
        self.register_buffer("_evecs_Vinv", None)
        self.register_buffer("_Kx_mean", None)
        self.N: int | None = None
        if variance_explained < 0. or variance_explained > 1.:
            raise ValueError("variance_explained must be in [0, 1]")
        self.variance_explained = variance_explained
        self.update_kernel_matrix(Kx_train_train)

    def _device(self) -> torch.device:
        """Return the active device for NLR cached buffers."""
        if self._evecs is not None:
            return self._evecs.device
        if self._Kx_mean is not None:
            return self._Kx_mean.device
        raise RuntimeError("NLR buffers are not initialized")

    def update_kernel_matrix(self, Kx_train_train: Tensor) -> None:
        """Recompute the low-rank kernel representation from training data.

        Args:
            Kx_train_train: Training kernel matrix with shape ``(N, N)``.
        """
        N = self.N = Kx_train_train.size(0)
        # Column means — reused for out-of-sample centering
        Kx_mean = Kx_train_train.mean(dim=0)
        # Double-centre the Gram matrix
        row_mean = Kx_train_train.mean(dim=1, keepdim=True)
        grand_mean = row_mean.mean()
        G = Kx_train_train - row_mean - Kx_mean.unsqueeze(0) + grand_mean
        # Eigendecomposition of the empirical covariance operator
        S = G / N
        evals, evecs = torch.linalg.eigh(S)
        evals = torch.flip(evals, dims=[0]).clamp(min=0.)
        evecs = torch.flip(evecs, dims=[1]).contiguous()
        # Truncate at variance_explained
        cumvar = torch.cumsum(evals, dim=0)
        target = self.variance_explained * cumvar[-1]
        n_components = min(int(torch.searchsorted(cumvar, target).item()) + 1, N)
        self._evals = evals[:n_components].contiguous()
        self._evecs = evecs[:, :n_components].contiguous()
        self._Kx_mean = Kx_mean
        # OLS weight matrix in spectral space: V @ diag(1 / (N * lambda))
        self._evecs_Vinv = self._evecs / (N * self._evals)   # N x D

    def predict(
            self,
            Kx_test_train: Tensor,          # M x N
            u_train: Tensor,                # M x N
    ) -> Tensor:                            # M
        """Predict responses at test points.

        Args:
            Kx_test_train: Cross-kernel between test and train, shape ``(M, N)``.
            u_train: Row-aligned training responses, shape ``(M, N)``.

        Returns:
            Tensor of shape ``(M,)``.
        """
        device = self._device()
        Kx_test_train = Kx_test_train.to(device)
        u_train = u_train.to(device)
        # Column-centre only — row-centering is a no-op because 1^T v = 0
        Kx_test_train = Kx_test_train - self._Kx_mean.unsqueeze(0)
        ubar = u_train.mean(dim=1)                           # M
        proj_kx = Kx_test_train @ self._evecs_Vinv           # M x D
        proj_u  = u_train @ self._evecs                      # M x D  (centering u is also a no-op)
        return ubar + (proj_kx * proj_u).sum(dim=1)

    def predict_weights(self, Kx_test_train: Tensor) -> Tensor:
        """Return the weight vector mapping ``u_train`` to ``predict``.

        Args:
            Kx_test_train: Cross-kernel between test and train, shape ``(M, N)``.

        Returns:
            Weight matrix ``W`` of shape ``(M, N)`` such that
            ``predict(K, u) == (W * u).sum(1)`` when ``u`` is ``(M, N)``, or
            ``W @ u`` when ``u`` is ``(N, K)`` (see
            ``pairwise_predict_through_weights``).
        """
        device = self._device()
        Kx_test_train = Kx_test_train.to(device)
        if Kx_test_train.ndim != 2 or Kx_test_train.size(1) != self.N:
            raise ValueError("Kx_test_train must have shape (M, N)")
        Kx_test_train = Kx_test_train - self._Kx_mean.unsqueeze(0)
        proj_kx   = Kx_test_train @ self._evecs_Vinv         # M x D
        W_centered = proj_kx @ self._evecs.T                  # M x N
        # evecs^T @ 1 = 0  =>  W_centered @ 1 = 0;
        # add 1/N uniformly to fold the intercept (mean) into the weights
        return W_centered + (1.0 / self.N)

    def predict_through_weights(
            self,
            Kx_test_train: Tensor,          # M x N
            u_train: Tensor,                # M x N
    ) -> Tensor:
        """Convenience wrapper: ``predict`` via the weight matrix."""
        W = self.predict_weights(Kx_test_train)
        return (W * u_train.to(W.device)).sum(dim=1)

    def pairwise_predict_through_weights(
            self,
            Kx_test_train: Tensor,          # M x N
            u_train: Tensor,                # N x K
    ) -> Tensor:
        """Compute predictions for K target sets simultaneously.

        Args:
            Kx_test_train: Cross-kernel, shape ``(M, N)``.
            u_train: Training responses for K targets, shape ``(N, K)``.

        Returns:
            Tensor of shape ``(M, K)``.
        """
        W = self.predict_weights(Kx_test_train)
        return W @ u_train.to(W.device)

    def __repr__(self) -> str:
        n_components = len(self._evals) if self._evals is not None else None
        return (
            f"NLR(N={self.N}, D={n_components}, "
            f"variance_explained={self.variance_explained})"
        )


class NLBR(torch.nn.Module):
    """Nonlinear Bayesian regression.

    U = a + b(x) + e
    a ~ Gaussian
    b ~ Gaussian

    a has mean induced by u_prior and variance proportional to the error variance
    b has mean induced by u_prior and covariance operator proportional to a power of
    the empirical covariance operator of the features

    The main purpose of this class is to compute the posterior predictive mean E[U|X=x,D].

    """

    def __init__(
            self,
            Kx_train_train: Tensor,
            intercept_variance_ratio: float = 0.,
            regression_variance_ratio: float = 1e-6,
            variance_explained: float = 0.99,
            power: float = 0.
    ):
        """Initialize NLBR and precompute its spectral state.

        Args:
            Kx_train_train: Training kernel matrix with shape `(N, N)`.
            intercept_variance_ratio: Ratio controlling intercept shrinkage.
            regression_variance_ratio: Ratio controlling regression strength.
            variance_explained: Fraction of cumulative variance to retain in
                the eigenspectrum (in `[0, 1]`).
            power: Exponent used in the spectral prior scaling.

        Raises:
            ValueError: If `variance_explained` is outside `[0, 1]`.
        """
        super().__init__()
        # Register tensors as buffers so module.to(device) migrates all cached state.
        self.register_buffer("_evals", None)
        self.register_buffer("_evecs", None)
        self.register_buffer("_evals_Vinv", None)
        self.register_buffer("_evals_bProj", None)
        self.register_buffer("_evals_VinvbProj", None)
        self.register_buffer("_evecs_Vinv", None)
        self.register_buffer("_evecs_bProj", None)
        self.register_buffer("_evecs_VinvbProj", None)
        self.register_buffer("_Kx_mean", None)
        self.N: int | None = None
        self.intercept_variance_ratio: float | Parameter | None = None
        self.regression_variance_ratio: float | Parameter | None = None
        # initialize
        if variance_explained < 0. or variance_explained > 1.:
            raise ValueError("variance_explained must be in [0, 1]")
        self.power = power
        self.variance_explained = variance_explained
        self.update_kernel_matrix(Kx_train_train)
        self.update_variance_ratios(intercept_variance_ratio, regression_variance_ratio)

    def _device(self) -> torch.device:
        """Return the active device for NLBR cached buffers."""
        if self._evecs is not None:
            return self._evecs.device
        if self._Kx_mean is not None:
            return self._Kx_mean.device
        raise RuntimeError("NLBR buffers are not initialized")

    def update_kernel_matrix(self, Kx_train_train: Tensor) -> None:
        """Recompute the low-rank kernel representation from training data.

        Args:
            Kx_train_train: Training kernel matrix with shape `(N, N)`.

        Side effects:
            Updates `N` and cached buffers used by `ese`.
        """
        N = self.N = Kx_train_train.size(0)
        # Column means — reused for out-of-sample centering in ese()
        Kx_mean = Kx_train_train.mean(dim=0)
        # Double-centre the Gram matrix in O(N^2) instead of the O(N^3) H @ K @ H einsum
        row_mean = Kx_train_train.mean(dim=1, keepdim=True)
        grand_mean = row_mean.mean()
        G = Kx_train_train - row_mean - Kx_mean.unsqueeze(0) + grand_mean
        # Compute covariance operator
        S = G / N
        # Compute eigendecomposition of covariance operator
        # eigh returns eigenvalues in ascending order; flip to descending in O(D)
        evals, evecs = torch.linalg.eigh(S)
        evals = torch.flip(evals, dims=[0]).clamp(min=0.)
        evecs = torch.flip(evecs, dims=[1]).contiguous()
        # Keep only up to variance explained (also roots out the zero eigenvalues)
        cumvar = torch.cumsum(evals, dim=0)
        totalvar = cumvar[-1]
        target = self.variance_explained * totalvar
        n_components = min(int(torch.searchsorted(cumvar, target).item()) + 1, N)
        self._evals = evals[:n_components].contiguous()
        self._evecs = evecs[:, :n_components].contiguous()
        self._Kx_mean = Kx_mean

    def update_variance_ratios(
            self,
            intercept_variance_ratio: float | Parameter | None = None,
            regression_variance_ratio: float | Parameter | None = None
    ) -> None:
        """Update intercept/regression variance ratios and derived buffers.

        Args:
            intercept_variance_ratio: Optional new intercept ratio.
            regression_variance_ratio: Optional new regression ratio.

        Raises:
            ValueError: If provided ratios are outside valid ranges.
        """
        if intercept_variance_ratio is not None:
            if intercept_variance_ratio < 0.:
                raise ValueError("intercept_variance_ratio must be nonnegative")
            self.intercept_variance_ratio = intercept_variance_ratio
        if regression_variance_ratio is not None:
            if regression_variance_ratio < 1e-10:
                raise ValueError("regression_variance_ratio must be positive")
            r2 = self.regression_variance_ratio = regression_variance_ratio
            evals = self._evals
            p = self.power
            if r2 < float("inf"):
                self._evals_Vinv = evals.pow(p) * r2 / (self.N * evals.pow(p + 1.) * r2 + 1.)
                self._evals_bProj = evals.pow(-1. - p) / (r2 * self.N)
            else:
                self._evals_Vinv = 1. / (self.N * evals)
                self._evals_bProj = torch.zeros_like(evals)
            self._evals_VinvbProj = self._evals_Vinv*self._evals_bProj
            # Precompute scaled eigenvectors once here to avoid per-call products in ese()
            self._evecs_Vinv = self._evecs * self._evals_Vinv   # N x D
            self._evecs_bProj = self._evecs * self._evals_bProj  # N x D
            self._evecs_VinvbProj = self._evecs * self._evals_VinvbProj  # N x D

    def ppm(
            self,
            Kx_test_train: Tensor,          # M x N
            u_train: Tensor,                # M x N
            u_prior: Tensor | None = None,  # M x N (optional prior)
    ) -> Tensor:
        """Posterior predictive mean computed directly.

        Args:
            Kx_test_train: Cross-kernel between test and train, shape `(M, N)`.
            u_train: Row-aligned train-side responses, shape `(M, N)`.
            u_prior: Optional prior mean with shape `(M, N)`.

        Returns:
            Tensor with shape `(M,)`.

        Notes:
            Inputs are moved to the module device before computation.
        """
        device = self._device()
        Kx_test_train = Kx_test_train.to(device)
        u_train = u_train.to(device)
        if u_prior is not None:
            u_prior = u_prior.to(device)

        # Compute centred feature matrix
        Kx_test_train = Kx_test_train - self._Kx_mean.unsqueeze(0)
        Kx_test_train = Kx_test_train - Kx_test_train.mean(dim=1, keepdim=True)
        # Centre u_train
        ubar = u_train.mean(dim=1)
        u_train = u_train - ubar.unsqueeze(-1)
        # Coordinate representation of the prior
        if u_prior is None:
            u_prior = torch.zeros_like(u_train)
        a = u_prior.mean(dim=1)
        u_prior = u_prior - a.unsqueeze(-1)
        b = (u_prior @ self._evecs) @ self._evecs_bProj.T  # M x N
        diff = u_train + b
        # Compute intercept
        r2 = self.intercept_variance_ratio
        if r2 < float("inf"):
            intercept = (a + self.N * r2 * ubar) / (1 + self.N * r2)
        else:
            intercept = ubar
        proj_kx = Kx_test_train @ self._evecs_Vinv  # M x D
        proj_diff = diff @ self._evecs              # M x D
        ip = (proj_kx * proj_diff).sum(dim=1)       # M
        return intercept + ip

    def ppm_weights(self, Kx_test_train: Tensor) -> tuple[Tensor, Tensor]:
        """Return weights mapping `u_train` and `u_prior` to `ppm`.

        Args:
            Kx_test_train: Cross-kernel between test and train, shape `(M, N)`.

        Returns:
            Weight matrices `W_train` and `W_prior` with shape `(M, N)`.
            These can be used to compute pairwise PPM given `K` target points:
            `W_train @ u_train + W_prior @ u_prior`, a `(M, K)` tensor, where `u_train` and `u_prior`
            are `(N, K)` tensors. They can also be used to compute the PPM efficiently at the `M` test points via
            `(W_train * u_train).sum(1) + (W_prior * u_prior).sum(1)`,
            where `u_train` and `u_prior` are `(M, N)` tensors.
            See `ppm_through_weights` and `pairwise_ppm_through_weights` for examples.
        """
        device = self._device()
        Kx_test_train = Kx_test_train.to(device)

        if Kx_test_train.ndim != 2 or Kx_test_train.size(1) != self.N:
            raise ValueError("Kx_test_train must have shape (M, N) with N equal to training size.")

        # Mirror ppm() centering of test-train features.
        Kx_test_train = Kx_test_train - self._Kx_mean.unsqueeze(0)
        Kx_test_train = Kx_test_train - Kx_test_train.mean(dim=1, keepdim=True)

        # Regression terms from centred features to centred responses.
        proj_kx = Kx_test_train @ self._evecs_Vinv              # M x D
        W_train_centered = proj_kx @ self._evecs.T              # M x N
        proj_kx_prior = Kx_test_train @ self._evecs_VinvbProj   # M x D
        W_prior_centered = proj_kx_prior @ self._evecs.T        # M x N

        # Fold centering and intercept into raw-u weights so they can be applied directly.
        r2 = self.intercept_variance_ratio
        if r2 < float("inf"):
            train_mean_coeff = (self.N * r2) / (1 + self.N * r2)
            prior_mean_coeff = 1 / (1 + self.N * r2)
        else:
            train_mean_coeff = 1.0
            prior_mean_coeff = 0.0

        train_mean_coeff = torch.as_tensor(train_mean_coeff, dtype=Kx_test_train.dtype, device=device)
        prior_mean_coeff = torch.as_tensor(prior_mean_coeff, dtype=Kx_test_train.dtype, device=device)

        W_train = W_train_centered + (train_mean_coeff - W_train_centered.sum(dim=1, keepdim=True)) / self.N
        W_prior = W_prior_centered + (prior_mean_coeff - W_prior_centered.sum(dim=1, keepdim=True)) / self.N

        return W_train, W_prior

    def ppm_through_weights(
            self,
            Kx_test_train: Tensor,          # M x N
            u_train: Tensor,                # M x N
            u_prior: Tensor | None = None,  # M x N (optional prior)
    ) -> Tensor:
        W_train, W_prior = self.ppm_weights(Kx_test_train)
        u_train = u_train.to(W_train.device)
        ppm_train = (W_train * u_train).sum(dim=1)
        if u_prior is None:
            return ppm_train
        else:
            u_prior = u_prior.to(W_train.device)
            ppm_prior = (W_prior * u_prior).sum(dim=1)
            return ppm_train + ppm_prior

    def pairwise_ppm_through_weights(
            self,
            Kx_test_train: Tensor,          # M x N
            u_train: Tensor,                # N x K
            u_prior: Tensor | None = None,  # N x K
    ) -> Tensor:
        W_train, W_prior = self.ppm_weights(Kx_test_train)
        u_train = u_train.to(W_train.device)
        ppm_train = W_train @ u_train
        if u_prior is None:
            return ppm_train
        else:
            u_prior = u_prior.to(W_train.device)
            ppm_prior = W_prior @ u_prior
            return ppm_train + ppm_prior

    def __repr__(self) -> str:
        n_components = len(self._evals) if self._evals is not None else None
        return (
            f"NLBR(N={self.N}, D={n_components}, power={self.power}, "
            f"variance_explained={self.variance_explained}, "
            f"intercept_variance_ratio={self.intercept_variance_ratio}, "
            f"regression_variance_ratio={self.regression_variance_ratio})"
        )
