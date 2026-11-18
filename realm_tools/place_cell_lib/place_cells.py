"""
place_cells.py

Vectorised place cell ensembles for the REALM / VPCE framework.

Two ensemble types share the same RBF core:

    SpatialPlaceCellEnsemble   — cells defined in 2D (x, y) space
                                  used by navigation / RL agents

    VisualPlaceCellEnsemble    — cells defined in high-dimensional feature
                                  space (the VPCE model)

Both expose the same interface:

    ensemble.activate(inputs)  →  np.ndarray  shape (N, K)

where N is the number of observations and K is the number of place cells.
Activations are computed with a single vectorised pass — no per-cell Python
loops, no mutable state stored inside cell objects.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def rbf_activations(features, centers, radii, normalize='row'):
    """
    Compute the RBF activation matrix.

    Uses the identity  ||a-b||² = ||a||² - 2a·b + ||b||²  to compute all
    pairwise distances via matrix multiplication, avoiding the O(N·K·D)
    memory cost of the naive broadcast approach.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
        Observation feature vectors.  For SpatialPlaceCellEnsemble D=2.
    centers  : np.ndarray, shape (K, D)
        Place cell centroids in feature space.
    radii    : np.ndarray, shape (K,)
        RBF sigma (receptive field width) per cell.
    normalize : {'row', 'minmax', None}
        How to normalise the raw RBF response.
        - 'row'    : each row sums to 1 (population code — default).
        - 'minmax' : per-cell min-max along the column so each cell's peak
                     response across the batch → 1 and its minimum → 0.
                     Only meaningful when features covers a range of
                     observations (e.g. a whole trajectory or arena sweep).
        - None     : raw Gaussian response in (0, 1].

    Returns
    -------
    np.ndarray, shape (N, K)
    """
    features = np.atleast_2d(features)
    f_sq     = np.sum(features ** 2, axis=1, keepdims=True)   # (N, 1)
    c_sq     = np.sum(centers  ** 2, axis=1)                   # (K,)
    fc       = features @ centers.T                             # (N, K)
    dist_sq  = np.maximum(f_sq - 2 * fc + c_sq, 0)            # (N, K)
    raw      = np.exp(-dist_sq / (2 * radii ** 2))             # (N, K)

    if normalize == 'row':
        row_sums = raw.sum(axis=1, keepdims=True)
        return raw / np.where(row_sums == 0, 1, row_sums)
    if normalize == 'minmax':
        col_min = raw.min(axis=0, keepdims=True)
        col_max = raw.max(axis=0, keepdims=True)
        span    = col_max - col_min
        return (raw - col_min) / np.where(span == 0, 1, span)
    if normalize is None:
        return raw
    raise ValueError(f"unknown normalize={normalize!r}; expected 'row', 'minmax', or None")


# ---------------------------------------------------------------------------
# Ensemble classes
# ---------------------------------------------------------------------------

class SpatialPlaceCellEnsemble:
    """
    Place cell ensemble defined in 2D (x, y) physical space.

    Replaces the old PlaceCell / PlaceCellNetwork pair with a fully
    vectorised implementation.  No per-cell objects, no mutable activity
    state — activation is computed on demand and returned as a numpy array.

    Parameters
    ----------
    centers : array-like, shape (K, 2)   — cell centres in metres
    radii   : array-like, shape (K,)     — RBF sigma per cell in metres

    Example
    -------
    >>> ensemble = SpatialPlaceCellEnsemble(centers, radii)
    >>> act = ensemble.activate(np.array([[x, y]]))   # (1, K)
    >>> act = ensemble.activate(positions)             # (N, K)
    """

    def __init__(self, centers, radii):
        self.centers = np.asarray(centers, dtype=np.float64)   # (K, 2)
        self.radii   = np.asarray(radii,   dtype=np.float64)   # (K,)

    @property
    def n_cells(self):
        return len(self.radii)

    def activate(self, positions, normalize='row'):
        """
        Compute activations for one or more (x, y) positions.

        Parameters
        ----------
        positions : array-like, shape (N, 2) or (2,)
        normalize : {'row', 'minmax', None} — see `rbf_activations`.

        Returns
        -------
        np.ndarray, shape (N, K)
        """
        return rbf_activations(
            np.asarray(positions, dtype=np.float64),
            self.centers,
            self.radii,
            normalize=normalize,
        )


class VisualPlaceCellEnsemble:
    """
    Visual Place Cell (VPCE) ensemble defined in high-dimensional feature space.

    Replaces the old VisiualPlaceCell / VisualPlaceCellNetwork pair.
    Centers and radii are produced by fit_kmeans / fit_gmm in clustering.py
    and are stored in the PlaceCellEnsemble HDF5 model file.

    Parameters
    ----------
    centers : array-like, shape (K, D)   — cluster centroids in feature space
    radii   : array-like, shape (K,)     — RBF sigma per cell

    Example
    -------
    >>> ensemble = VisualPlaceCellEnsemble(centers, radii)
    >>> act = ensemble.activate(features_pca)   # (N, K)
    >>> act = ensemble.activate(single_obs)     # (1, K)
    """

    def __init__(self, centers, radii, channels=None, scales=None):
        self.centers = np.asarray(centers, dtype=np.float64)   # (K, D)
        self.radii   = np.asarray(radii,   dtype=np.float64)   # (K,)
        # How the feature space was built. Carried so a saved ensemble can be
        # applied to a NEW observation later -- see `encode`.
        self.channels = tuple(channels) if channels is not None else None
        self.scales   = (np.asarray(scales, dtype=np.float64)
                         if scales is not None else None)

    @property
    def n_cells(self):
        return len(self.radii)

    @property
    def feature_dim(self):
        return self.centers.shape[1]

    # ----------------------------------------------------------- persistence

    def save(self, filename, method=None, maze=None):
        """Write the ensemble to an HDF5 file (PlaceCellEnsemble format)."""
        # Imported here, not at module scope: this module is pure numpy and
        # gets imported inside Webots controllers, where pulling h5py in for
        # an ensemble that is only ever constructed in memory is dead weight.
        from realm_tools.experiment_lib.loggers.place_cell_model import PlaceCellEnsemble
        model = PlaceCellEnsemble()
        model.centers, model.radii = self.centers, self.radii
        model.save(filename, method=method, maze=maze,
                   channels=self.channels, scales=self.scales)
        return filename

    @classmethod
    def load(cls, filename):
        """Read an ensemble written by `save`, ready to `activate`."""
        from realm_tools.experiment_lib.loggers.place_cell_model import PlaceCellEnsemble
        model = PlaceCellEnsemble.load(filename)
        return cls(model.centers, model.radii,
                   channels=model.channels, scales=model.scales)

    # ----------------------------------------------------------- new input

    def encode(self, blocks, normalize='row'):
        """Activations for new observations given their raw feature blocks.

        `blocks` is the dict that `channels.load_channel_blocks` returns, or
        any mapping of the same block names to (N, d) arrays. The stored
        channel order and RMS divisors are reapplied, so the result lands in
        the same space as the centroids.

        This is the method to use on new input. Calling `activate` with a
        freshly assembled matrix instead silently re-measures the divisors
        from whatever batch is at hand, which rescales the observation
        relative to the centroids and quietly corrupts every activation.
        """
        from realm_tools.place_cell_lib.channels import assemble
        if self.channels is None or self.scales is None:
            raise ValueError(
                'this ensemble carries no channel configuration, so new input '
                'cannot be encoded into its feature space. It was saved before '
                'channels/scales were recorded -- refit it, or assemble the '
                'features yourself with the original scales and call activate().')
        X = assemble(blocks, self.channels, scales=self.scales)
        if X.shape[1] != self.feature_dim:
            raise ValueError(f'assembled {X.shape[1]} features, '
                             f'but the ensemble has {self.feature_dim}')
        return self.activate(X, normalize=normalize)

    def activate(self, features, normalize='row'):
        """
        Compute activations for one or more feature vectors.

        Parameters
        ----------
        features : array-like, shape (N, D) or (D,)
        normalize : {'row', 'minmax', None} — see `rbf_activations`.

        Returns
        -------
        np.ndarray, shape (N, K)
        """
        return rbf_activations(
            np.asarray(features, dtype=np.float64),
            self.centers,
            self.radii,
            normalize=normalize,
        )
