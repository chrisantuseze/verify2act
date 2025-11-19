import types
import torch
import numpy as np
import pytest

from relational_dynamics.config.base_config import BaseConfig
from relational_dynamics.base_RD import RelationalDynamics


def make_minimal_args():
    # Create a minimal args object with required attributes used in RelationalDynamics
    a = types.SimpleNamespace()
    # dataset/IO args
    a.result_dir = '/tmp'
    a.use_tensorboard = False
    a.checkpoint_path = ''
    a.use_multiple_train_dataset = False
    a.use_multiple_test_dataset = False
    a.pick_place = False
    a.pushing = False
    a.set_max = False
    a.max_objects = 3
    a.online_planning = False
    a.start_id = 0
    a.max_size = 0
    a.start_test_id = 0
    a.test_max_size = 0
    a.updated_behavior_params = False
    a.save_data_path = ''
    a.evaluate_new = False
    a.evaluate_pickplace = False
    a.using_multi_step_statistics = False
    a.total_sub_step = 1
    a.use_shared_latent_embedding = False
    a.use_seperate_latent_embedding = False
    a.push_3_steps = False
    a.POMDP_push = False
    a.sudo_pickplace = False
    a.push_steps = 1
    a.single_step_training = True
    a.add_noise_pc = False
    a.train_object_identity = False
    a.use_rgb = False
    a.use_boundary_relations = False
    a.consider_z_offset = False
    a.seperate_env_id = False
    a.max_env_num = 0
    a.env_first_step = 0
    a.use_discrete_z = False
    a.fast_training = True
    a.one_bit_env = False
    a.rcpe = False
    a.pe = False
    a.relation_angle = 45
    a.bookshelf_env_shift = 0
    a.lfd_search = False
    a.get_hidden_label = False
    a.get_inside_relations = False
    a.enable_place_inside = False
    a.binary_grasp = False
    a.open_close_drawer = False
    a.softmax_identity = False
    a.train_inside_feasibility = False
    a.use_discrete_place = False
    a.seperate_place = False
    a.enable_leap_num = False
    a.batch_feasibility = False
    # paths expected by DataLoader
    a.train_dir = None
    a.test_dir = None

    # model args
    a.node_emb_size = 16
    a.n_layers = 1
    a.n_heads = 1
    a.d_hidden = 16
    a.z_dim = 4
    a.simple_encoding = False
    a.transformer_dynamics = False
    a.seperate_discrete_continuous = False
    a.torch_embedding = False
    a.complicated_pre_dynamics = False
    a.direct_transformer = False
    a.enable_high_push = False
    a.enable_place_inside = False
    a.use_discrete_place = False
    a.latent_discrete_continuous = False
    a.seperate_action_emb = False
    a.dim_feedforward = 16
    a.use_mlp_encoder = False
    a.pose_num = 3
    a.train_env_identity = False
    a.train_grasp_identity = False
    a.train_inside_feasibility = False
    a.binary_grasp = False
    a.seperate_identity = False
    a.one_bit_env = False
    a.train_obj_move = False
    a.train_obj_boundary = False
    a.transformer_decoder = False
    a.remove_orientation = False
    a.pose_trans_decoder = False

    # training args
    a.emb_lr = 1e-3
    a.learning_rate = 1e-3
    a.sqrt_var = 1.0
    a.delta_forward = False
    a.latent_forward = False

    return a


@pytest.fixture
def rd():
    args = make_minimal_args()
    config = BaseConfig(args)
    rd = RelationalDynamics(config)
    return rd


def make_fake_batch(args):
    # Create a minimal single-scene batch element compatible with process_data
    num_objects = 3
    num_points = 8
    point_dim = 3
    # voxel list: (num_nodes, num_points, point_dim)
    voxels = np.random.rand(num_objects, num_points, point_dim).astype(np.float32)

    data = {
        'num_objects': num_objects,
        'action': torch.zeros((1, 1, args.max_objects + 3 + 1)),
        'relation': torch.zeros((1, num_objects*(num_objects-1), args.z_dim)),
        'one_hot_encoding': torch.zeros((1, num_objects, args.max_objects + 3)),
        'all_action_label': np.zeros((1,)),
        'all_object_pair_voxels_single': torch.tensor(voxels),
        'env_identity': torch.zeros((1, num_objects, 3)),
        'all_gt_grapable_list': torch.zeros((1, num_objects, 1)),
        'all_6DOF_pose_fast': torch.zeros((1, num_objects, 6)),
        'edge_attr': torch.LongTensor(list(zip(*[(i,j) for i in range(num_objects) for j in range(num_objects) if i!=j]))),
        'all_obj_boundaty': None,
        'position': None,
        'quaternian': None,
        'extents': None,
        'all_hidden_tensor': None,
        'new_latent': None,
        'support_suface_id': [np.zeros((1,1), dtype=np.int64)],
        'buffer_tensor_0': [torch.zeros((1, args.max_objects + 3))],
        'all_hidden_label': None,
    }
    return data


def test_compute_time_step_outputs(rd):
    # build small batch
    data = make_fake_batch(rd.config.args)
    batch = [data]
    x = rd.process_data(batch)

    total_steps = 1
    rd.edge_index = x['batch_edge_attr']
    outputs = rd._compute_time_step_outputs(x, total_steps)

    # outputs is a tuple of five lists
    assert isinstance(outputs, tuple) and len(outputs) == 5
    latent_list = outputs[0]
    assert len(latent_list) == total_steps


def test_get_action_embeddings(rd):
    data = make_fake_batch(rd.config.args)
    batch = [data]
    x = rd.process_data(batch)
    rd.edge_index = x['batch_edge_attr']

    total_steps = 1
    outputs = rd._compute_time_step_outputs(x, total_steps)
    (current_latent, discrete_action, continuous_action, current_action_continuous, current_action, skill_label) = rd._get_action_embeddings(x, 0, outputs[0])

    # check tensor types and shapes
    assert current_latent.shape[1] == rd.config.args.max_objects
    assert discrete_action.dim() == 3
    assert continuous_action.dim() == 3
    assert current_action_continuous.dim() == 3

