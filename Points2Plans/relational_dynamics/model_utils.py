import torch
import numpy as np


def process_data(config, batch_data, push_3_steps = False): 
    '''Process raw batch data and collect relevant objects in a dict.

    Input: batch_data is a list of per-scene dicts produced by the DataLoader.
    Output: x_dict contains tensors organized for training: stacked along
    batch dimension (and sequence/time dimension where applicable).
    Values set to -1 in the data indicate masked / missing entries and are
    respected later during loss computation.
    '''
    args = config.args
    x_dict = dict()

    
    # assert(len(batch_data) == 1)

    x_dict['batch_num_objects'] = []
    x_dict['batch_action'] = []
    x_dict['batch_all_obj_pair_relation'] = []
    x_dict['batch_one_hot_encoding'] = []
    x_dict['batch_skill_label'] = []
    x_dict['batch_voxel_list_single'] = []
    x_dict['batch_env_identity'] = []
    x_dict['batch_grasp_identity'] = []
    x_dict['batch_6DOF_pose'] = []
    x_dict['support_suface_id'] = []
    x_dict['buffer_tensor_0'] = []

    # Stack items from every element in batch_data into lists, then convert
    # to tensors (or keep as lists where appropriate). The DataLoader
    # already formats keys like 'all_object_pair_voxels_single' so we keep
    # the same naming and shapes expected by downstream code.
    for b, data in enumerate(batch_data):
        x_dict['batch_num_objects'].append(data['num_objects'])

        x_dict['batch_action'].append(data['action'])
        x_dict['batch_all_obj_pair_relation'].append(data['relation'])
        x_dict['batch_one_hot_encoding'].append(data['one_hot_encoding'])

        x_dict['batch_skill_label'].append(data['all_action_label'])
        x_dict['batch_voxel_list_single'].append(data['all_object_pair_voxels_single'])
        x_dict['batch_env_identity'].append(data['env_identity'])
        x_dict['batch_grasp_identity'].append(data['all_gt_grapable_list'])
        x_dict['batch_6DOF_pose'].append(data['all_6DOF_pose_fast'])
        x_dict['batch_edge_attr'] = data['edge_attr']
        x_dict['batch_obj_boundary'] = data['all_obj_boundaty']
        x_dict['batch_position'] = data['position']
        x_dict['batch_quaternian'] = data['quaternian']
        x_dict['batch_extents'] = data['extents']
        x_dict['batch_all_hidden_tensor'] = data['all_hidden_tensor']
        x_dict['new_latent'] = data['new_latent']
        x_dict['support_suface_id'].append(data['support_suface_id'])
        x_dict['buffer_tensor_0'].append(data['buffer_tensor_0'])
        x_dict['batch_all_hidden_label'] = data['all_hidden_label']

    # Convert lists to stacked tensors for efficient batch processing
    x_dict['batch_env_identity'] = torch.squeeze(torch.stack(x_dict['batch_env_identity']), 1)
    x_dict['batch_all_obj_pair_relation'] = torch.squeeze(torch.stack(x_dict['batch_all_obj_pair_relation']), 1)
    x_dict['batch_voxel_list_single'] = torch.squeeze(torch.stack(x_dict['batch_voxel_list_single']), 1)
    x_dict['batch_action'] = torch.squeeze(torch.stack(x_dict['batch_action']), 1)
    x_dict['batch_skill_label'] = np.squeeze(np.stack(x_dict['batch_skill_label']), 1)
    x_dict['batch_one_hot_encoding'] = torch.squeeze(torch.stack(x_dict['batch_one_hot_encoding']), 1)
    x_dict['batch_grasp_identity'] = torch.squeeze(torch.stack(x_dict['batch_grasp_identity']), 1)
    x_dict['batch_6DOF_pose'] = torch.squeeze(torch.stack(x_dict['batch_6DOF_pose']), 1)

    return x_dict

def process_data_plan(config, batch_data, push_3_steps = False): 
    '''Process data for planning (assumes batch size == 1).

    This is similar to `process_data` but keeps a single scene and does not
    stack across a batch dimension. The planner expects single-scene input.
    '''
    args = config.args
    x_dict = dict()

    # Planner operates on single-scene inputs
    assert(len(batch_data) == 1)

    for b, data in enumerate(batch_data):
        x_dict['batch_num_objects'] = data['num_objects']
        x_dict['batch_action'] = data['action']
        x_dict['batch_all_obj_pair_relation'] = data['relation']
        x_dict['batch_one_hot_encoding'] = data['one_hot_encoding']
        x_dict['batch_edge_attr'] = data['edge_attr']
        x_dict['batch_skill_label'] = data['all_action_label']
        x_dict['batch_voxel_list_single'] = data['all_object_pair_voxels_single']
        x_dict['batch_env_identity'] = data['env_identity']
        x_dict['batch_grasp_identity'] = data['all_gt_grapable_list']
        x_dict['batch_6DOF_pose'] = data['all_6DOF_pose_fast']
        x_dict['batch_obj_boundary'] = data['all_obj_boundaty']

        x_dict['batch_position'] = data['position']
        x_dict['batch_quaternian'] = data['quaternian']
        x_dict['batch_extents'] = data['extents']
        x_dict['batch_all_hidden_tensor'] = data['all_hidden_tensor']
        x_dict['new_latent'] = data['new_latent']

        x_dict['support_suface_id'] = data['support_suface_id']
        x_dict['buffer_tensor_0'] = data['buffer_tensor_0']

        x_dict['batch_all_hidden_label'] = data['all_hidden_label']

    return x_dict

def build_masks(x_tensor_dict):
    """Construct commonly used masks for training loss computation.

    Returns a dict with masks keyed by name.
    """
    masks = {}
    masks['relational_mask'] = (x_tensor_dict['batch_all_obj_pair_relation']==-1)
    masks['env_mask'] = (x_tensor_dict['batch_env_identity']==-1)
    masks['object_level_mask'] = masks['env_mask'][:, :, :, [0]]
    masks['latent_space_mask'] = masks['object_level_mask'].repeat(1,1,1,256)
    masks['graspable_mask'] = (x_tensor_dict['batch_grasp_identity']==-1)
    masks['position_mask'] = (x_tensor_dict['batch_6DOF_pose']==-1)
    return masks

def collision_check_2D(bounding_box_1, bounding_box_2):
    bbox1_left = bounding_box_1[0][0]
    bbox1_right = bounding_box_1[2][0]
    bbox1_top = bounding_box_1[0][1]
    bbox1_bottom = bounding_box_1[1][1]

    bbox2_left = bounding_box_2[0][0]
    bbox2_right = bounding_box_2[2][0]
    bbox2_top = bounding_box_2[0][1]
    bbox2_bottom = bounding_box_2[1][1]

    # standard axis-aligned bbox overlap check
    if (bbox1_left < bbox2_right and
        bbox1_right > bbox2_left and
        bbox1_top < bbox2_bottom and
        bbox1_bottom > bbox2_top):
        return True
    else:
        return False