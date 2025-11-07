from __future__ import annotations

import torch

__all__ = [
    "normalize_vector",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "quaternion_to_frame",
    "frame_to_quaternion",
    "apply_rotation",
]


def normalize_vector(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalise a batch of vectors.

    Args:
        vector: Arbitrary tensor that ends with a 3-vector axis. Shape ``[..., 3]``.
        eps: Minimum norm used to avoid division-by-zero.

    Returns:
        Normalised vectors with the same shape as the input.
    """
    return vector / (vector.norm(dim=-1, keepdim=True).clamp_min(eps))


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternions into rotation matrices.

    Args:
        quaternion: Unit quaternions in ``[..., 4]`` order ``(qx, qy, qz, qw)``.

    Returns:
        Rotation matrices with shape ``[..., 3, 3]``.
    """
    qx, qy, qz, qw = quaternion.unbind(-1)
    # Quadratic terms reused across matrix entries.
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    m00 = 1 - 2 * (yy + zz)
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)

    m10 = 2 * (xy + wz)
    m11 = 1 - 2 * (xx + zz)
    m12 = 2 * (yz - wx)

    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = 1 - 2 * (xx + yy)

    # Stack rows in column-major fashion to build a 3x3 matrix per quaternion.
    matrix = torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ],
        dim=-2,
    )
    return matrix


def matrix_to_quaternion(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert rotation matrices back to unit quaternions.

    Args:
        matrix: Rotation matrices in shape ``[..., 3, 3]``.
        eps: Numerical guard to prevent square-roots of negative values.

    Returns:
        Unit quaternions shaped ``[..., 4]`` with order ``(qx, qy, qz, qw)``.
    """
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]

    trace = m00 + m11 + m22
    # Branch-free conversion keeps gradients stable.
    qw = torch.sqrt(torch.clamp(trace + 1.0, min=eps)) / 2.0
    qx = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=eps)) / 2.0
    qy = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=eps)) / 2.0
    qz = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=eps)) / 2.0

    # Recover the correct signs for each component via the skew-symmetric entries.
    qx = torch.copysign(qx, matrix[..., 2, 1] - matrix[..., 1, 2])
    qy = torch.copysign(qy, matrix[..., 0, 2] - matrix[..., 2, 0])
    qz = torch.copysign(qz, matrix[..., 1, 0] - matrix[..., 0, 1])

    quat = torch.stack([qx, qy, qz, qw], dim=-1)
    quat = normalize_vector(quat)

    # Quaternions have a sign ambiguity; canonicalise by forcing qw >= 0 so the
    # round-trip matches the original representation.
    sign = torch.where(quat[..., -1:] < 0, -1.0, 1.0)
    return quat * sign


def quaternion_to_frame(
    quaternion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode a quaternion into explicit basis vectors.

    Args:
        quaternion: Tensor of quaternions with shape ``[..., 4]``.

    Returns:
        Tuple ``(forward, right, up)`` where each entry has shape ``[..., 3]``.
    """
    rot = quaternion_to_matrix(quaternion)
    # Columns of the rotation matrix give us the rotated canonical basis.
    forward = rot[..., 0]  # Local +X axis points along the gripper jaw.
    right = rot[..., 1]    # Local +Y axis.
    up = rot[..., 2]       # Local +Z axis (typically "up" from the gripper palm).
    return forward, right, up


def frame_to_quaternion(
    forward: torch.Tensor, up: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Reconstruct a quaternion from two basis vectors.

    Args:
        forward: Forward axes ``[..., 3]``.
        up: Approximate up axes ``[..., 3]`` (need not be orthogonal).
        eps: Small constant used while normalising.

    Returns:
        Quaternions in ``[..., 4]`` order ``(qx, qy, qz, qw)``.
    """
    forward_unit = normalize_vector(forward, eps)
    # Gram-Schmidt: remove forward component from the supplied up vector.
    up_proj = up - (forward_unit * up).sum(dim=-1, keepdim=True) * forward_unit
    up_unit = normalize_vector(up_proj, eps)
    right_unit = normalize_vector(torch.cross(up_unit, forward_unit, dim=-1), eps)

    # Columns of this matrix are the orthonormal basis vectors.
    rot = torch.stack([forward_unit, right_unit, up_unit], dim=-1)  # [..., 3, 3]
    return matrix_to_quaternion(rot)


def apply_rotation(rotation: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """
    Apply batched rotation matrices to vectors.

    Args:
        rotation: Rotation matrices with shape ``[..., 3, 3]``.
        vectors: Vectors to be rotated in shape ``[..., 3]``.

    Returns:
        Rotated vectors with the same leading dimensions as the inputs.
    """
    return torch.einsum("...ij,...j->...i", rotation, vectors)
