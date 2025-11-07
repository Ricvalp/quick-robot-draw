import torch

from src.policies.platonic_policy import (
    _components_to_pose,
    _pack_components,
    _pose_to_components,
    _unpack_components,
)
from src.utils.geometry import matrix_to_quaternion, quaternion_to_matrix


def random_quaternion(batch_shape: tuple[int, ...] = ()) -> torch.Tensor:
    """Sample random unit quaternions for the requested batch shape."""
    q = torch.randn(*batch_shape, 4)
    return q / q.norm(dim=-1, keepdim=True)


def test_quaternion_matrix_round_trip():
    """Ensure quaternion -> matrix -> quaternion returns the same orientation."""
    q = random_quaternion((16,))
    matrix = quaternion_to_matrix(q)  # (16, 3, 3)
    q_round = matrix_to_quaternion(matrix)  # (16, 4)
    canonical = torch.where(q[..., -1:] < 0, -q, q)  # enforce qw >= 0
    diff_direct = (canonical - q_round).abs().max()
    diff_flipped = (canonical + q_round).abs().max()
    assert min(diff_direct, diff_flipped) < 3e-5


def test_pose_feature_round_trip():
    """The pose <-> feature conversion used by the policy should be lossless."""
    position = torch.randn(2, 3, 3)
    quaternion = random_quaternion((2, 3))
    gripper = torch.rand(2, 3, 1)
    pose = torch.cat([position, quaternion, gripper], dim=-1)  # (2, 3, 8)

    grasp, orientation, position_comp = _pose_to_components(pose)
    recovered = _components_to_pose(grasp, orientation, position_comp)

    # Account for quaternion sign ambiguity before comparing.
    canonical = pose.clone()
    sign = torch.where(canonical[..., 6:7] < 0, -1.0, 1.0)
    canonical[..., 3:7] = canonical[..., 3:7] * sign

    assert torch.allclose(canonical[..., :3], recovered[..., :3], atol=1e-5)
    assert torch.allclose(canonical[..., 7:], recovered[..., 7:], atol=1e-5)

    orig_matrix = quaternion_to_matrix(canonical[..., 3:7])
    rec_matrix = quaternion_to_matrix(recovered[..., 3:7])
    assert torch.allclose(orig_matrix, rec_matrix, atol=1e-5)


def test_pack_unpack_symmetry():
    """Packing action features must be perfectly invertible."""
    grasp = torch.randn(4, 6, 1)
    orientation = torch.randn(4, 6, 2, 3)
    position = torch.randn(4, 6, 3)
    packed = _pack_components(grasp, orientation, position)
    unpack_grasp, unpack_orientation, unpack_position = _unpack_components(packed)
    assert torch.allclose(grasp, unpack_grasp)
    assert torch.allclose(orientation, unpack_orientation)
    assert torch.allclose(position, unpack_position)
