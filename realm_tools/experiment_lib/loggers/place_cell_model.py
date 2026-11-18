import os
import numpy as np
import h5py


class PlaceCellEnsemble:
    def __init__(self):
        self.centers  = []   # (N_cells, feature_dim)
        self.radii    = []   # (N_cells,)
        self.method   = None
        self.maze     = None
        # How the feature vectors these centers live in were built. Without
        # these, a saved ensemble cannot be applied to a new observation:
        # see `scales` in channels.assemble.
        self.channels = None  # tuple[str] -- block names, in order
        self.scales   = None  # (n_channels,) -- per-block RMS divisor

    def save(self, filename, method=None, maze=None, channels=None, scales=None):
        """
        Save the place cell ensemble to an HDF5 file.
        Creates the directory if it does not exist.

        Structure:
            centers      (N_cells x feature_dim)  -- cluster centroids in feature space
            radii        (N_cells,)               -- RBF sigma per cell
            scales       (n_channels,)           -- per-block RMS divisor, optional
            attrs:
                method   str   e.g. 'kmeans', 'gmm' or 'agglomerative'
                maze     str   e.g. 'circ_lm8_r10'
                n_cells  int
                channels str   comma-joined block names, in concatenation order

        `channels` and `scales` record how the feature space was built, and
        are what let a later session encode a NEW observation into the same
        space. They are optional so that ensembles written before they
        existed still load.
        """
        if not filename.endswith('.h5'):
            filename += '.h5'
        dir_name = os.path.dirname(filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with h5py.File(filename, 'w') as f:
            f.create_dataset('centers', data=np.array(self.centers), compression='gzip')
            f.create_dataset('radii',   data=np.array(self.radii))
            f.attrs['n_cells'] = len(self.radii)
            if method is not None:
                f.attrs['method'] = method
            if maze is not None:
                f.attrs['maze'] = maze
            channels = channels if channels is not None else self.channels
            scales   = scales   if scales   is not None else self.scales
            if channels is not None:
                # Comma-joined rather than a string array: block names never
                # contain commas, and this reads identically from h5dump and
                # from any h5py version.
                f.attrs['channels'] = ','.join(channels)
            if scales is not None:
                f.create_dataset('scales', data=np.asarray(scales, dtype=np.float64))

        print(f"Place cell ensemble saved to {filename}  ({len(self.radii)} cells)")

    @staticmethod
    def load(filename):
        """
        Load a place cell ensemble from an HDF5 file.
        """
        if not filename.endswith('.h5'):
            filename += '.h5'
        ensemble = PlaceCellEnsemble()
        with h5py.File(filename, 'r') as f:
            ensemble.centers = f['centers'][:]
            ensemble.radii   = f['radii'][:]
            ensemble.method  = f.attrs.get('method', None)
            ensemble.maze    = f.attrs.get('maze', None)
            ch = f.attrs.get('channels', None)
            ensemble.channels = tuple(ch.split(',')) if ch else None
            ensemble.scales   = f['scales'][:] if 'scales' in f else None
        return ensemble
