import torch
from torch import Tensor


def lift_scalars(scalars: Tensor, group) -> Tensor:
    """
    Broadcast scalar features across all group elements and flatten the result.

    Args:
        scalars: Tensor of shape [B, N, C_s].
        group:  Platonic group reference supplying the rotation frames.

    Returns:
        Tensor of shape [B, N, G * C_s].
    """
    if scalars is None:
        raise ValueError("Scalar features must be provided when calling lift_scalars.")
    if scalars.ndim != 3:
        raise ValueError("Expected scalar features with shape [B, N, C_s].")

    B, N, C_s = scalars.shape
    return scalars.unsqueeze(2).repeat(1, 1, group.G, 1)  # [B, N, G, C_s]


def lift_vectors(vectors: Tensor, group) -> Tensor:
    """
    Rotate vector features into every group frame and flatten the result.

    Args:
        vectors: Tensor of shape [B, N, C_v, D].
        group:  Platonic group reference supplying the rotation frames.

    Returns:
        Tensor of shape [B, N, G * C_v * D].
    """
    if vectors is None:
        raise ValueError("Vector features must be provided when calling lift_vectors.")
    if vectors.ndim != 4:
        raise ValueError("Expected vector features with shape [B, N, C_v, D].")

    frames = group.elements.to(dtype=vectors.dtype, device=vectors.device)  # [G, D, D]
    rotated = torch.einsum(
        "gij,bncj->bngci", frames, vectors
    )  # [B, N, G, C_v, D]
    return rotated.flatten(-2, -1)  # [B, N, G, C_v * D]


def readout_scalars(x: Tensor, group) -> Tensor:
    """Average across the group axis to recover scalar channels."""
    return x.mean(dim=-2)  # [..., G, C_s] -> [..., C_s]


def readout_vectors(x: Tensor, group) -> Tensor:
    """Rotate vectors back to the canonical frame and average the group axis."""
    x = x.unflatten(-1, (-1, group.dim))  # [..., G, C_v, D]
    frames = group.elements.to(dtype=x.dtype, device=x.device)  # [G, D, D]
    projected = torch.einsum("gji,...gcj->...ci", frames, x)  # [..., C_v, D]
    return projected / group.G  # [..., C_v, D]


def lift(scalars: Tensor | None, vectors: Tensor | None, group) -> Tensor:
    """
    Combine scalar and vector features into a single group-aware representation.

    Args:
        scalars: Optional tensor [B, N, C_s].
        vectors: Optional tensor [B, N, C_v, D].
        group:   Platonic group reference.

    Returns:
        Tensor of shape [B, N, G * (C_s + C_v * D)].
    """
    components = []
    if scalars is not None:
        components.append(lift_scalars(scalars, group))  # [B, N, G, C_s]
    if vectors is not None:
        components.append(lift_vectors(vectors, group))  # [B, N, G, C_v*D]
    if not components:
        raise ValueError("At least one of scalars or vectors must be provided to lift.")
    return torch.cat(components, dim=-1).flatten(-2, -1)  # [B, N, G*(C_s + C_v*D)]


def to_scalars_vectors(
    x: Tensor, num_scalars: int, num_vectors: int, group
) -> tuple[Tensor, Tensor]:
    """
    Split a group-aware representation back into scalar and vector pieces.

    Args:
        x:            Tensor shaped [..., G * (num_scalars + num_vectors * D)].
        num_scalars:  Number of scalar channels.
        num_vectors:  Number of vector channels.
        group:        Platonic group reference.

    Returns:
        Tuple ``(scalars, vectors)`` with shapes ``[..., num_scalars]`` and
        ``[..., num_vectors, D]`` respectively.
    """
    grouped = x.unflatten(-1, (group.G, -1))  # [..., G, C_total]
    scalars, vectors = grouped.split(
        [num_scalars, num_vectors * group.dim], dim=-1
    )  # [..., G, num_scalars], [..., G, num_vectors * D]

    scalars_out = readout_scalars(scalars, group)  # [..., num_scalars]
    vectors_out = readout_vectors(vectors, group)  # [..., num_vectors, D]
    return scalars_out, vectors_out
