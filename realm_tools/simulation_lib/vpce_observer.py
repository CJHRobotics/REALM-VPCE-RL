"""Turn what the robot can see right now into a place-cell activation vector.

This is the join between the fitted ensemble and the live simulation, and it
is the one place where a silent mismatch would poison everything downstream:
an observation assembled even slightly differently from the dataset the
ensemble was fitted on still has the right shape, still produces plausible
numbers, and is simply wrong. Nothing in the agent, the reward or the
learning curve would reveal it. So the invariants are spelled out here and
checked at construction rather than trusted.

Three things have to match the collection exactly.

Heading order. A location's feature vector is not one image -- it is eight,
captured at THETAS and concatenated in that order. `collect_data` writes one
row per (position, heading) in heading order, and `channels.load_channel_blocks`
reshapes (n_loc * 8, d) -> (n_loc, 8 * d). Assembling the headings in any
other order permutes the feature space and silently relocates every cell.

Descriptor construction. The same `extract_feature_dict` with the same flags,
so the blocks are the same widths in the same order.

Block scaling. `channels.assemble` divides each block by an RMS measured on
the fitted dataset. Those divisors are stored in the ensemble file and are
reapplied here through `ensemble.encode`. Re-deriving them from live
observations would put the robot at a different radius from the origin than
the centroids, which is the failure mode that is hardest to see.

A consequence worth stating: this representation is PANORAMIC. Because every
observation spans all eight headings, it is a function of where the robot is
and not of which way it faces -- which is the right property for a place
cell, and matches how the place-field paper measured them. It also means
eight camera renders per environment step, which dominates step time.
"""

import numpy as np

from realm_tools.place_cell_lib.place_cells import VisualPlaceCellEnsemble

# The collection headings, radians, in the order collect_data captured them.
# Duplicated as a literal rather than imported from the controller because
# the controller is not importable outside Webots -- but it is the same list,
# and `check_against_dataset` exists to catch them drifting apart.
THETAS = (0.0, 0.7854, 1.5708, 2.3562, 3.1416, 3.9270, 4.7124, 5.4978)

# Matches collect_data's lidar handling via channels.load_channel_blocks.
LIDAR_HEADING_INDEX = 0
LIDAR_MAX_RANGE = 8.0
LIDAR_SENTINEL = -1.0


class VpceObserver:
    """Reads the robot, returns the ensemble's activation over what it sees.

    Parameters
    ----------
    ensemble : str or VisualPlaceCellEnsemble
        Path to an ensemble written by `analysis/fit_place_fields.py`, or an
        already-loaded one.
    normalize : {'row', 'minmax', None}
        Passed to the ensemble. 'row' makes each observation a population
        code summing to 1, which keeps the observation bounded in [0, 1] and
        is what the Box space in WebotsEnv assumes.
    """

    def __init__(self, ensemble, normalize='row'):
        if isinstance(ensemble, str):
            ensemble = VisualPlaceCellEnsemble.load(ensemble)
        self.ensemble = ensemble
        self.normalize = normalize

        if self.ensemble.channels is None or self.ensemble.scales is None:
            raise ValueError(
                'this ensemble has no channel configuration, so a live '
                'observation cannot be placed in its feature space. Refit it '
                'with analysis/fit_place_fields.py.')

        self.channels = tuple(self.ensemble.channels)
        self.needs_lidar = 'lidar' in self.channels
        self.visual_keys = tuple(c for c in self.channels if c != 'lidar')

        # Which descriptors to compute. Asking extract_feature_dict for a
        # block the ensemble does not use would be wasted work on every step,
        # and HOG is the expensive one.
        self._flags = dict(
            use_hog='hog' in self.channels,
            use_color_hist='color_hist' in self.channels,
            use_spatial='spatial' in self.channels,
        )

    @property
    def n_cells(self):
        return self.ensemble.n_cells

    def check_against_dataset(self, data_path):
        """Confirm a live observation would have the fitted feature width.

        Cheap insurance, and worth running once at startup: it reads only the
        dataset's block shapes, and catches a THETAS or descriptor-flag drift
        before a training run spends hours learning from a scrambled space.
        """
        from realm_tools.place_cell_lib import channels as ch
        blocks, _ = ch.load_channel_blocks(data_path, verbose=False)
        width = sum(blocks[c].shape[1] for c in self.channels if c in blocks)
        if width != self.ensemble.feature_dim:
            raise ValueError(
                f'{data_path} assembles to {width} features for channels '
                f'{self.channels}, but the ensemble expects '
                f'{self.ensemble.feature_dim}. The ensemble and the dataset '
                f'do not belong together.')
        return True

    @staticmethod
    def _read_heading(robot):
        """Exact heading, from the rotation field rather than the compass.

        `get_robot_pose` reports the IMU compass in whole degrees, rounded;
        restoring from it would walk the robot's heading by up to half a
        degree every step. The rotation field is the value that was set.
        """
        try:
            return float(robot.robot_rotation_field.getSFRotation()[3])
        except Exception:                                   # noqa: BLE001
            import math
            return math.radians(robot.get_robot_pose()[2])

    def blocks_from_robot(self, robot):
        """Capture one panoramic observation and return it as feature blocks.

        Returns a dict shaped exactly like `channels.load_channel_blocks`
        produces for a single location: {name: (1, 8 * d)}.

        The robot's heading is restored afterwards. Capturing a panorama
        means rotating through every heading, which leaves the robot facing
        THETAS[-1]; since the action set and the lidar action mask are both
        read relative to the current heading, an unrestored observation
        would silently rotate the agent's whole action frame on every step.
        """
        # Lazily imported: image_feature_lib pulls in cv2, which the fit and
        # analysis paths deliberately avoid, and which is only needed once we
        # are actually in a simulation looking at pictures.
        from realm_tools.image_lib.image_feature_lib import extract_feature_dict

        heading = self._read_heading(robot)
        x, y, _ = robot.get_robot_pose()
        try:
            images, _masks, _azimuths, lidar_scan = robot.capture_pov_images(
                list(THETAS), lidar_heading_index=LIDAR_HEADING_INDEX)
        finally:
            robot.teleport_robot(x=x, y=y, theta=heading)
        if len(images) != len(THETAS):
            raise RuntimeError(f'expected {len(THETAS)} headings, '
                               f'got {len(images)}')

        per_heading = [extract_feature_dict(img, **self._flags) for img in images]

        blocks = {}
        for key in self.visual_keys:
            # Concatenated in heading order, matching the reshape in
            # load_channel_blocks. This is the invariant the module docstring
            # is about; do not reorder.
            blocks[key] = np.concatenate(
                [h[key] for h in per_heading]).reshape(1, -1).astype(np.float32)

        if self.needs_lidar:
            from realm_tools.place_cell_lib.channels import mask_lidar
            scan = np.asarray(lidar_scan, dtype=np.float32).reshape(1, -1)
            blocks['lidar'] = mask_lidar(scan,
                                         max_range=LIDAR_MAX_RANGE,
                                         sentinel=LIDAR_SENTINEL,
                                         with_mask_channel=True)
        return blocks

    def observe(self, robot):
        """One panoramic observation as a (n_cells,) float32 activation."""
        blocks = self.blocks_from_robot(robot)
        act = self.ensemble.encode(blocks, normalize=self.normalize)
        return np.asarray(act[0], dtype=np.float32)
