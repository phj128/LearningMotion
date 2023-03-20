import numpy as np
import torch

from .utils import normalize, denormalize


def split_features(
    data,
    start_pose=0,
    start_pose_inv=264,
    start_trajectory=330,
    start_contact=447,
    start_traj_inv=452,
    start_goal=504,
    start_interaction=647,
):
    pose = data[:, start_pose:start_pose_inv]
    pose_inv = data[:, start_pose_inv:start_trajectory]
    traj = data[:, start_trajectory:start_contact]
    contact = data[:, start_contact:start_traj_inv]

    traj_inv = data[:, start_traj_inv:start_goal]
    goal = data[:, start_goal:start_interaction]
    if data.shape[1] > start_interaction:
        interaction = data[:, start_interaction:]
    else:
        interaction = None
    return pose, pose_inv, traj, contact, traj_inv, goal, interaction


def normalize_vector(v):
    # norm_v = v.pow(2).sum(dim=-1).sqrt().unsqueeze(-1)
    norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)
    return v / norm_v


def split_pose_feat(data):
    pos = data[:, :, :3].clone()
    forward = data[:, :, 3:6].clone()
    up = data[:, :, 6:9].clone()
    if data.shape[-1] > 9:
        velocity = data[:, :, 9:12].clone()
    else:
        velocity = None
    return pos, forward, up, velocity


def project_root(pose_mat_world):
    """_summary_
        return the matrix of root

    Args:
        pose_mat_world (tensor): [B, J, 4, 4]

    Returns:
        mat_world_root (tensor): [B, 4, 4]
    """

    bs = pose_mat_world.shape[0]
    pose_pos_world = pose_mat_world[:, :, :3, 3]

    lastrow = torch.tensor(
        np.tile(np.array([[[0, 0, 0, 1]]]), (bs, 1, 1)),
        device=pose_mat_world.device,
        dtype=torch.float32,
    )
    up_root = torch.tensor(
        np.tile(np.array([[0, 1, 0]]), (bs, 1)),
        device=pose_mat_world.device,
        dtype=torch.float32,
    )

    pelvis_pos = pose_pos_world[:, 0, :3]
    pos_root = project_pos_on_plane(pelvis_pos)

    # We use 4 joints to define the forward direction,
    # including leftHip, rightHip, leftShoulder and rightShoulder,
    # whose index is 1, 5, 13, and 19

    ############################
    #
    #
    # TODO: Implement here!
    #
    #
    ############################

    mat_world_root = torch.cat([mat_world_root, lastrow], dim=1)
    return mat_world_root


def project_pos_on_plane(pos):
    pos_xz = pos.clone()
    pos_xz[:, 1] = 0.0
    return pos_xz


def transform_mat(mat, root_mat_in, root_mat_out_inv):
    mat_world = torch.matmul(root_mat_in[:, None], mat)
    # Convert to transforms wrt frame i+1
    mat_transformed = torch.matmul(root_mat_out_inv, mat_world)
    return mat_transformed


def convert_vel(velocity_out, root_mat_in, root_mat_out_inv, nj=22):
    """_summary_

    Args:
        velocity_out (tensor): [B, nj, 3]
        root_mat_in (tensor): [B, 4, 4]_
        root_mat_out_inv (tensor): [B, 4, 4]
        nj (int, optional): number of joints. Defaults to 22.

    Returns:
        volocity_out_transformed: (tensor) [B, nj, 4]
    """

    ############################
    #
    #
    # TODO: Implement here!
    #
    #
    ############################

    return velocity_out_transformed


###############################################################################################################
#################### TRAJECTRORY
###############################################################################################################


def traj_mat_2_vec(mat, style, window_size=13, num_actions=5):
    bs = mat.shape[0]
    traj = torch.zeros(
        (bs, window_size, 4 + num_actions), device=mat.device, dtype=mat.dtype
    )
    traj[:, :, 0] = mat[..., 0, 3]
    traj[:, :, 1] = mat[..., 2, 3]
    traj[:, :, 2] = mat[..., 0, 2]
    traj[:, :, 3] = mat[..., 2, 2]
    traj[:, :, 4:] = style
    return traj.reshape(bs, window_size * (4 + num_actions))


def traj_vec_2_mat(data, window_size=13):
    # Get position and direction data only, ignore style data
    bs = data.shape[0]
    lastrow = torch.tensor(
        np.tile(np.array([[[0, 0, 0, 1]]]), (bs, window_size, 1, 1)),
        device=data.device,
        dtype=torch.float32,
    )
    pos_dir = data.reshape(bs, window_size, -1)[:, :, :4]
    style = data.reshape(bs, window_size, -1)[:, :, 4:]

    # Convert to 3D data
    pos = torch.zeros(
        (pos_dir.shape[0], pos_dir.shape[1], 3), device=data.device, dtype=torch.float32
    )
    pos[:, :, 0] = pos_dir[:, :, 0]
    pos[:, :, 2] = pos_dir[:, :, 1]

    forward = torch.zeros(
        (pos_dir.shape[0], pos_dir.shape[1], 3), device=data.device, dtype=torch.float32
    )
    forward[:, :, 0] = pos_dir[:, :, 2]
    forward[:, :, 2] = pos_dir[:, :, 3]

    forward = normalize_vector(forward)
    up = torch.tensor(
        np.tile(np.array([[0, 1, 0]]), (bs, window_size, 1)),
        device=data.device,
        dtype=torch.float32,
    )
    up = normalize_vector(up)
    right = torch.cross(forward, up)
    right = normalize_vector(right)

    mat = torch.stack([right, up, forward, pos], dim=-1)
    mat = torch.cat([mat, lastrow], dim=2)
    return mat, style


def transform_traj(traj_vec, root_mat_in, root_mat_out_inv, num_actions=5):
    traj_mat, traj_style = traj_vec_2_mat(traj_vec)
    traj_transformed = transform_mat(traj_mat, root_mat_in, root_mat_out_inv)
    traj_transformed = traj_mat_2_vec(
        traj_transformed, traj_style, num_actions=num_actions
    )
    return traj_transformed


###############################################################################################################
#################### GOAL Related
###############################################################################################################
def goal_vec_2_mat(data, window_size=13):
    # Get position and direction data only, ignore style data
    bs = data.shape[0]
    lastrow = torch.tensor(
        np.tile(np.array([[[0, 0, 0, 1]]]), (bs, window_size, 1, 1)),
        device=data.device,
        dtype=torch.float32,
    )
    pos_dir = data.reshape(bs, window_size, -1)[:, :, :6]
    style = data.reshape(bs, window_size, -1)[:, :, 6:]

    pos = pos_dir[:, :, :3]
    forward = pos_dir[:, :, 3:]
    forward = normalize_vector(forward)
    up = torch.tensor(
        np.tile(np.array([[0, 1, 0]]), (bs, window_size, 1)),
        device=data.device,
        dtype=torch.float32,
    )
    up = normalize_vector(up)
    right = torch.cross(forward, up)
    right = normalize_vector(right)

    mat = torch.stack([right, up, forward, pos], dim=-1)
    mat = torch.cat([mat, lastrow], dim=2)
    return mat, style


def goal_mat_2_vec(mat, style, window_size=13, num_actions=5):
    bs = mat.shape[0]
    traj = torch.zeros(
        (bs, window_size, 6 + num_actions), dtype=torch.float32, device=mat.device
    )
    traj[:, :, :3] = mat[..., :3, 3]
    traj[:, :, 3:6] = mat[..., :3, 2]
    traj[:, :, 6:] = style
    return traj.reshape(bs, window_size * (6 + num_actions))


def transform_goal(goal_vec, root_mat_in, root_mat_out_inv, num_actions=5):
    goal_mat, goal_style = goal_vec_2_mat(goal_vec)
    goal_transformed = transform_mat(goal_mat, root_mat_in, root_mat_out_inv)
    goal_transformed = goal_mat_2_vec(
        goal_transformed, goal_style, num_actions=num_actions
    )
    return goal_transformed


###############################################################################################################
#################### Pose
###############################################################################################################


def pose_vec_2_mat(pose):
    bs = pose.shape[0]
    nj = pose.shape[1]
    lastrow = torch.tensor(
        np.tile(np.array([[[[0, 0, 0, 1]]]]), (bs, nj, 1, 1)),
        device=pose.device,
        dtype=pose.dtype,
    )
    pos, forward, up, velocity = split_pose_feat(pose)
    # Get input joint transforms wrt frame i
    forward = normalize_vector(forward)
    up = normalize_vector(up)
    right = torch.cross(forward, up)
    right = normalize_vector(right)

    mat = torch.stack([right, up, forward, pos], dim=-1)
    mat = torch.cat([mat, lastrow], dim=2)
    return mat, velocity


def pose_mat_2_vec(mat, velocity):
    # Get velocities and convert them to velocities wrt frame i+1
    pose_transformed = [
        mat[..., :3, 3],
        mat[..., :3, 2],
        mat[..., :3, 1],
        velocity[..., :3],
    ]
    pose_transformed = torch.cat(pose_transformed, dim=-1)
    return pose_transformed


def transform_pose(pose_vec_in, pose_vec_out, NJ=22, pose_feat_dim=12):
    """_summary_
    Transform pose that is relative to frame i to the one that is relative to frame i + 1

    Args:
        pose_vec_in (tensor): [B, C]
        pose_vec_out (tensor): [B, C]
        NJ (int, optional): The number of joints. Defaults to 22.
        pose_feat_dim (int, optional): The dim of each joint. Defaults to 12.

    Returns:
        transformed_feature: [B, C]
    """
    pose_vec_in = pose_vec_in.reshape(-1, NJ, pose_feat_dim)
    pose_vec_out = pose_vec_out.reshape(-1, NJ, pose_feat_dim)
    bs = pose_vec_in.shape[0]

    pose_in_mat, _ = pose_vec_2_mat(pose_vec_in)
    # # Get root transforms at frame i
    root_mat_in = project_root(pose_in_mat)

    pose_mat_out, velocity_out = pose_vec_2_mat(pose_vec_out)

    pose_mat_world_out = torch.matmul(root_mat_in[:, None], pose_mat_out)
    # Get root transform for frame i+1
    root_mat_out = project_root(pose_mat_world_out)

    # Convert to transforms wrt frame i+1
    root_mat_out_inv = torch.inverse(root_mat_out[:, None])
    pose_mat_out_transformed = torch.matmul(root_mat_out_inv, pose_mat_world_out)

    # Convert velocity
    velocity_out_transformed = convert_vel(
        velocity_out, root_mat_in, root_mat_out_inv, NJ
    )

    pose_vec_out_transformed = pose_mat_2_vec(
        pose_mat_out_transformed, velocity_out_transformed
    )

    return (
        pose_vec_out_transformed.reshape(bs, NJ * pose_feat_dim),
        root_mat_in,
        root_mat_out_inv,
    )


###############################################################################################################
#################### Inv Pose
###############################################################################################################


def get_last_traj_world(traj_vec, root_mat_in):
    traj_mat, _ = traj_vec_2_mat(traj_vec)
    traj_mat_world = torch.matmul(root_mat_in[:, None], traj_mat)
    return traj_mat_world[:, -1, :, :]


def transform_pose_inv(pose_inv_out, root_mat_in, traj_in, traj_out, NJ=22):
    ############################
    #
    #
    # TODO: Implement here!
    #
    #
    ############################

    return pose_inv_out_transformed


###############################################################################################################
#################### Inv Traj
###############################################################################################################
def inv_traj_mat_2_vec(mat, window_size=13):
    bs = mat.shape[0]
    traj = torch.zeros((bs, window_size, 4), device=mat.device, dtype=mat.dtype)
    traj[:, :, 0] = mat[..., 0, 3]
    traj[:, :, 1] = mat[..., 2, 3]
    traj[:, :, 2] = mat[..., 0, 2]
    traj[:, :, 3] = mat[..., 2, 2]
    return traj.reshape(bs, window_size * 4)


def inv_traj_vec_2_mat(data, window_size=13):
    # Get position and direction data only, ignore style data
    bs = data.shape[0]
    lastrow = torch.tensor(
        np.tile(np.array([[[0, 0, 0, 1]]]), (bs, window_size, 1, 1)),
        device=data.device,
        dtype=torch.float32,
    )
    pos_dir = data.reshape(bs, window_size, -1)[:, :, :4]

    # Convert to 3D data
    pos = torch.zeros(
        (pos_dir.shape[0], pos_dir.shape[1], 3), device=data.device, dtype=torch.float32
    )
    pos[:, :, 0] = pos_dir[:, :, 0]
    pos[:, :, 2] = pos_dir[:, :, 1]

    forward = torch.zeros(
        (pos_dir.shape[0], pos_dir.shape[1], 3), device=data.device, dtype=torch.float32
    )
    forward[:, :, 0] = pos_dir[:, :, 2]
    forward[:, :, 2] = pos_dir[:, :, 3]

    forward = normalize_vector(forward)
    up = torch.tensor(
        np.tile(np.array([[0, 1, 0]]), (bs, window_size, 1)),
        device=data.device,
        dtype=torch.float32,
    )
    up = normalize_vector(up)
    right = torch.cross(forward, up)
    right = normalize_vector(right)

    # check normalization above
    mat = torch.stack([right, up, forward, pos], dim=-1)
    mat = torch.cat([mat, lastrow], dim=2)
    return mat


def get_pivot_goal_world(goal_vec, root_mat_in):
    goal_mat, _ = goal_vec_2_mat(goal_vec)
    goal_mat_world = torch.matmul(root_mat_in[:, None], goal_mat)
    return goal_mat_world[:, 6, :, :]


def transform_traj_inv(traj_inv_out, root_mat_in, goal_in, goal_out):
    FoR_in = get_pivot_goal_world(goal_in, root_mat_in)
    FoR_out_inv = torch.inverse(get_pivot_goal_world(goal_out, root_mat_in)[:, None])

    traj_inv_out_mat = inv_traj_vec_2_mat(traj_inv_out)

    traj_inv_transformed = transform_mat(traj_inv_out_mat, FoR_in, FoR_out_inv)
    traj_inv_transformed = inv_traj_mat_2_vec(traj_inv_transformed)

    return traj_inv_transformed


###############################################################################################################
####################  Main
###############################################################################################################


def transform_data(inputs, outputs, num_actions=5, **kwargs):
    """_summary_

    Args:
        inputs (tensor): [B, C]
        outputs (tensor): [B, C]
        num_actions (int, optional): action numbers. Defaults to 5.

    Returns:
        _tensor: [B, C]
    """
    (
        pose_in,
        pose_inv_in,
        traj_in,
        contact_in,
        traj_inv_in,
        goal_in,
        interaction_in,
    ) = split_features(inputs)
    (
        pose_out,
        pose_inv_out,
        traj_out,
        contact_out,
        traj_inv_out,
        goal_out,
        interaction_out,
    ) = split_features(outputs)
    # Transform egocentric features
    pose_out_transformed, root_mat_in, root_mat_out_inv = transform_pose(
        pose_in, pose_out
    )
    traj_out_trasformed = transform_traj(
        traj_out, root_mat_in, root_mat_out_inv, num_actions=num_actions
    )
    goal_out_transformed = transform_goal(
        goal_out, root_mat_in, root_mat_out_inv, num_actions=num_actions
    )
    # inv
    pose_inv_out_transformed = transform_pose_inv(
        pose_inv_out, root_mat_in, traj_in, traj_out
    )
    traj_inv_out_transformed = transform_traj_inv(
        traj_inv_out, root_mat_in, goal_in, goal_out
    )

    outputs_transformed = torch.cat(
        [
            pose_out_transformed,
            pose_inv_out_transformed,
            traj_out_trasformed,
            contact_out,
            traj_inv_out_transformed,
            goal_out_transformed,
        ],
        dim=-1,
    )

    return outputs_transformed


def transform_output(
    p_prev,
    I,
    y_hat,
    input_mean,
    input_std,
    output_mean,
    output_std,
    state_dim=647,
    num_actions=5,
    **kwargs
):
    # denormalize the input
    inputs = denormalize(torch.cat((p_prev, I), dim=-1), input_mean, input_std)
    # denormalize the output
    outputs = denormalize(y_hat.reshape(-1, state_dim), output_mean, output_std)
    # transform the denormed data
    outputs_transformed = transform_data(inputs, outputs, num_actions=num_actions)
    # return normalized data
    return normalize(
        outputs_transformed, input_mean[..., :state_dim], input_std[..., :state_dim]
    )
