import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    """Solve linear assignment problem using Hungarian algorithm.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Matrix of shape (num_tracks, num_detections) containing association
        costs. A smaller cost indicates a better match.

    Returns
    -------
    np.ndarray
        Array of (track_index, detection_index) pairs with optimal assignment.
        If either side is empty, an empty array is returned.
    """

    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return np.asarray(list(zip(row_ind, col_ind)), dtype=int)


def tlbr_to_xyah(tlbr):
    """Convert [x1, y1, x2, y2] box to [cx, cy, aspect, height]."""
    x1, y1, x2, y2 = tlbr
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    a = w / (h + 1e-6)
    return np.array([cx, cy, a, h])


def xyah_to_tlbr(xyah):
    """Convert [cx, cy, aspect, height] box to [x1, y1, x2, y2]."""
    cx, cy, a, h = xyah
    w = a * h
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2])


class Detection:
    """Lightweight detection structure used by DeepSort."""

    def __init__(self, bbox, feature):
        self.tlbr = np.asarray(bbox, dtype=float)
        self.bbox = tlbr_to_xyah(self.tlbr)
        # Normalize feature for cosine distance
        feature = np.asarray(feature, dtype=float)
        self.feature = feature / (np.linalg.norm(feature) + 1e-6)


class KalmanFilter:
    """Simple Kalman filter for bounding box tracking."""

    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        covariance = np.eye(8)
        covariance[4:, 4:] *= 1000.0
        covariance *= 10.0
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = self._std_weight_position * mean[3]
        std_vel = self._std_weight_velocity * mean[3]
        motion_cov = np.diag(
            [
                std_pos,
                std_pos,
                std_pos,
                std_pos,
                std_vel,
                std_vel,
                std_vel,
                std_vel,
            ]
        ) ** 2

        mean = np.dot(self._motion_mat, mean)
        covariance = (
            self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        )
        return mean, covariance

    def project(self, mean, covariance):
        std = self._std_weight_position * mean[3]
        innovation_cov = np.diag([std, std, std, std]) ** 2
        mean = self._update_mat @ mean
        covariance = (
            self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        )
        return mean, covariance

    def update(self, mean, covariance, measurement):
        proj_mean, proj_cov = self.project(mean, covariance)
        kalman_gain = covariance @ self._update_mat.T @ np.linalg.inv(proj_cov)
        innovation = measurement - proj_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ proj_cov @ kalman_gain.T
        return new_mean, new_cov


class Track:
    """Internal track state used by DeepSort."""

    def __init__(self, mean, covariance, track_id, feature):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.features = [feature]
        self.time_since_update = 0
        self.hits = 1

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.time_since_update += 1

    def update(self, kf, detection):
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.bbox)
        self.features.append(detection.feature)
        self.time_since_update = 0
        self.hits += 1

    def to_tlbr(self):
        return xyah_to_tlbr(self.mean[:4])

    @property
    def feature(self):
        return self.features[-1]


class DeepSort:
    """DeepSort tracker optimized for low-latency operation."""

    def __init__(self, max_age=30, n_init=3, max_cosine_distance=0.2):
        self.max_age = max_age
        self.n_init = n_init
        self.max_cosine_distance = max_cosine_distance

        self.tracks = []
        self._next_id = 1
        self.kf = KalmanFilter()

    def _cosine_distance(self, tracks, detections):
        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((len(tracks), len(detections)))
        track_features = np.array([t.feature for t in tracks])
        det_features = np.array([d.feature for d in detections])
        cost = 1.0 - np.dot(track_features, det_features.T)
        return cost

    def _match(self, detections):
        cost_matrix = self._cosine_distance(self.tracks, detections)
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(self.tracks)), np.arange(len(detections))

        matches = linear_assignment(cost_matrix)
        unmatched_tracks = []
        unmatched_dets = []
        for t in range(len(self.tracks)):
            if t not in matches[:, 0]:
                unmatched_tracks.append(t)
        for d in range(len(detections)):
            if d not in matches[:, 1]:
                unmatched_dets.append(d)

        good_matches = []
        for t, d in matches:
            if cost_matrix[t, d] > self.max_cosine_distance:
                unmatched_tracks.append(t)
                unmatched_dets.append(d)
            else:
                good_matches.append((t, d))
        return good_matches, unmatched_tracks, unmatched_dets

    def update(self, detections):
        """Run one update step with new detections."""
        # Predict new locations of existing tracks
        for track in self.tracks:
            track.predict(self.kf)

        # Associate detections to tracks
        matches, unmatched_tracks, unmatched_dets = self._match(detections)

        # Update matched tracks with assigned detections
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[det_idx])

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            mean, cov = self.kf.initiate(det.bbox)
            track = Track(mean, cov, self._next_id, det.feature)
            self.tracks.append(track)
            self._next_id += 1

        # Age and remove unmatched tracks
        alive_tracks = []
        for idx, track in enumerate(self.tracks):
            if idx in unmatched_tracks:
                track.time_since_update += 1
            if track.time_since_update <= self.max_age:
                alive_tracks.append(track)
        self.tracks = alive_tracks

        # Output tracks that are confirmed
        outputs = []
        for track in self.tracks:
            if track.hits >= self.n_init and track.time_since_update == 0:
                outputs.append((track.track_id, track.to_tlbr(), track.feature))
        return outputs
