import multiprocessing as mp
import os
import time

import numpy as np
import torch
from progressbar import ProgressBar
from torch.autograd import Variable
from torch.utils.data import Dataset
from socket import gethostname

from physics_engine import RopeEngine, SoftEngine, SwimEngine
from physics_engine import sample_init_p_flight

from utils import rand_float, rand_int, calc_dis
from utils import init_stat, combine_stat, load_data, store_data


# ======================================================================================================================
def normalize(data, stat, var=False):
    for i in range(len(stat)):
        stat[i][stat[i][:, 1] == 0, 1] = 1.0
    if var:
        for i in range(len(stat)):
            s = Variable(torch.FloatTensor(stat[i]).to(data[i].device))
            data[i] = (data[i] - s[:, 0]) / s[:, 1]
    else:
        for i in range(len(stat)):
            data[i] = (data[i] - stat[i][:, 0]) / stat[i][:, 1]
    return data


def denormalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            s = Variable(torch.FloatTensor(stat[i])).to(data[i].device)
            data[i] = data[i] * s[:, 1] + s[:, 0]
    else:
        for i in range(len(stat)):
            data[i] = data[i] * stat[i][:, 1] + stat[i][:, 0]
    return data


# ======================================================================================================================

def prepare_input(data, stat, args, param=None, var=False):
    if args.env == 'Rope':
        data = normalize(data, stat, var)
        attrs, states, actions = data

        # print('attrs', attrs.shape, np.mean(attrs), np.std(attrs))
        # print('states', states.shape, np.mean(states), np.std(states))
        # print('acts', acts.shape, np.mean(actions), np.std(actions))

        N = len(attrs)

        # print('N', N)

        rel_attrs = np.zeros((N, N, args.relation_dim))

        '''relation #0 self: root <- root'''
        rel_attrs[0, 0, 0] = 1

        '''relation #1 spring: root <- child'''
        rel_attrs[0, 1, 1] = 1

        '''relation #2 spring: child <- root'''
        rel_attrs[1, 0, 2] = 1

        '''relation #3 spring bihop: root <- child'''
        rel_attrs[0, 2, 3] = 1

        '''relation #4 spring bihop: child <- root'''
        rel_attrs[2, 0, 4] = 1

        '''relation #5 spring: child <- child'''
        for i in range(1, N - 1):
            rel_attrs[i, i + 1, 5] = rel_attrs[i + 1, i, 5] = 1

        '''relation #6 spring bihop: child <- child'''
        for i in range(1, N - 2):
            rel_attrs[i, i + 2, 6] = rel_attrs[i + 2, i, 6] = 1

        '''relation #7 self: child <- child'''
        np.fill_diagonal(rel_attrs[1:, 1:, 7], 1)

        assert (rel_attrs.sum(2) <= 1).all()

        # check the number of each edge type
        rel_type_sum = np.sum(rel_attrs, axis=(0, 1))
        assert rel_type_sum[0] == 1
        assert rel_type_sum[1] == 1
        assert rel_type_sum[2] == 1
        assert rel_type_sum[3] == 1
        assert rel_type_sum[4] == 1
        assert rel_type_sum[5] == (N - 2) * 2
        assert rel_type_sum[6] == (N - 3) * 2
        assert rel_type_sum[7] == N - 1

    elif args.env in ['Soft', 'Swim']:
        init_p = param[3]
        data = normalize(data, stat, var)
        attrs, states, actions = data

        # print('attrs', attrs.shape, np.mean(attrs), np.std(attrs))
        # print('states', states.shape, np.mean(states), np.std(states))
        # print('acts', actions.shape, np.mean(actions), np.std(actions))

        N = len(attrs)

        # print('N', N)
        rel_attrs = np.zeros((N, N, args.relation_dim))

        num_spacial_rel_type = 9
        num_box_type = 3 if args.env == 'Swim' else 4

        for i in range(N):
            # normalized attributes
            type_i = np.where(attrs[i] > 0)[0][0]
            type_id = type_i
            # type_id = type_i * num_box_type + type_i
            rel_attrs[i, i, type_id * num_spacial_rel_type + 0] = 1  # self

            for j in range(N):
                if i == j:
                    continue

                delta = init_p[i, :2] - init_p[j, :2]

                assert (np.abs(delta) > 0).any()

                if (np.abs(delta) > 1).any():
                    # no contact
                    continue

                """
                get i and j type 
                Soft: [0: soft actuator, 1: soft, 2: rigid, 3: fixed]
                Swim: [0: soft actuator, 1: soft, 2: rigid]
                """
                # normalized attributes
                type_i = np.where(attrs[i] > 0)[0][0]
                type_id = type_i

                if np.sum(np.abs(delta)) == 1:
                    # contact at a corner
                    if delta[0] == 1:
                        rel_attrs[i, j, 1 + type_id * num_spacial_rel_type] = 1
                    elif delta[0] == -1:
                        rel_attrs[i, j, 2 + type_id * num_spacial_rel_type] = 1
                    elif delta[1] == 1:
                        rel_attrs[i, j, 3 + type_id * num_spacial_rel_type] = 1
                    elif delta[1] == -1:
                        rel_attrs[i, j, 4 + type_id * num_spacial_rel_type] = 1

                elif np.sum(np.abs(delta)) == 2:
                    # contact at a side
                    if delta[0] == 1 and delta[1] == 1:
                        rel_attrs[i, j, 5 + type_id * num_spacial_rel_type] = 1
                    elif delta[0] == 1 and delta[1] == -1:
                        rel_attrs[i, j, 6 + type_id * num_spacial_rel_type] = 1
                    elif delta[0] == -1 and delta[1] == 1:
                        rel_attrs[i, j, 7 + type_id * num_spacial_rel_type] = 1
                    elif delta[0] == -1 and delta[1] == -1:
                        rel_attrs[i, j, 8 + type_id * num_spacial_rel_type] = 1
                    else:
                        raise AssertionError(
                            "Unknown contact pattern %d %d" % (delta[0], delta[1]))
                else:
                    raise AssertionError(
                        "Unknown contact pattern %d %d" % (delta[0], delta[1]))

    else:
        raise AssertionError("unsupported env")

    return attrs, states, actions, rel_attrs


def gen_Rope(info):
    thread_idx, data_dir, data_names = info['thread_idx'], info['data_dir'], info['data_names']
    n_rollout, time_step = info['n_rollout'], info['time_step']
    dt, video, args, phase = info['dt'], info['video'], info['args'], info['phase']

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

    attr_dim = args.attr_dim  # root, child
    state_dim = args.state_dim  # x, y, xdot, ydot
    action_dim = args.action_dim
    param_dim = args.param_dim  # n_ball, init_x, k, damping, gravity

    act_scale = 2.
    ret_scale = 1.

    # attr, state, action
    stats = [init_stat(attr_dim), init_stat(state_dim), init_stat(action_dim)]

    engine = RopeEngine(dt, state_dim, action_dim, param_dim)

    group_size = args.group_size
    sub_dataset_size = n_rollout * args.num_workers // args.n_splits
    print('group size', group_size, 'sub_dataset_size', sub_dataset_size)
    assert n_rollout % group_size == 0
    assert args.n_rollout % args.n_splits == 0

    bar = ProgressBar()
    for i in bar(range(n_rollout)):
        rollout_idx = thread_idx * n_rollout + i
        group_idx = rollout_idx // group_size
        sub_idx = rollout_idx // sub_dataset_size

        num_obj_range = args.num_obj_range if phase in {'train', 'valid'} else args.extra_num_obj_range
        num_obj = num_obj_range[sub_idx]

        rollout_dir = os.path.join(data_dir, str(rollout_idx))

        param_file = os.path.join(data_dir, str(group_idx) + '.param')

        os.system('mkdir -p ' + rollout_dir)

        if rollout_idx % group_size == 0:
            engine.init(param=(num_obj, None, None, None, None))
            torch.save(engine.get_param(), param_file)
        else:
            while not os.path.isfile(param_file):
                time.sleep(0.5)
            param = torch.load(param_file)
            engine.init(param=param)

        for j in range(time_step):
            states_ctl = engine.get_state()[0]
            act_t = np.zeros((engine.num_obj, action_dim))
            act_t[0, 0] = (np.random.rand() * 2 - 1.) * act_scale - states_ctl[0] * ret_scale

            engine.set_action(action=act_t)

            states = engine.get_state()
            actions = engine.get_action()

            n_obj = engine.num_obj

            pos = states[:, :2].copy()
            vec = states[:, 2:].copy()

            '''reset velocity'''
            if j > 0:
                vec = (pos - states_all[j - 1, :, :2]) / dt

            if j == 0:
                attrs_all = np.zeros((time_step, n_obj, attr_dim))
                states_all = np.zeros((time_step, n_obj, state_dim))
                actions_all = np.zeros((time_step, n_obj, action_dim))

            '''attrs: [1, 0] => root; [0, 1] => child'''
            assert attr_dim == 2
            attrs = np.zeros((n_obj, attr_dim))
            # category: the first ball is fixed
            attrs[0, 0] = 1
            attrs[1:, 1] = 1

            assert np.sum(attrs[:, 0]) == 1
            assert np.sum(attrs[:, 1]) == engine.num_obj - 1

            attrs_all[j] = attrs
            states_all[j, :, :2] = pos
            states_all[j, :, 2:] = vec
            actions_all[j] = actions

            data = [attrs, states_all[j], actions_all[j]]

            store_data(data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))

            engine.step()

        datas = [attrs_all.astype(np.float64), states_all.astype(np.float64), actions_all.astype(np.float64)]

        for j in range(len(stats)):
            stat = init_stat(stats[j].shape[0])
            stat[:, 0] = np.mean(datas[j], axis=(0, 1))[:]
            stat[:, 1] = np.std(datas[j], axis=(0, 1))[:]
            stat[:, 2] = datas[j].shape[0]
            stats[j] = combine_stat(stats[j], stat)

    return stats


def gen_Soft(info):
    thread_idx, data_dir, data_names = info['thread_idx'], info['data_dir'], info['data_names']
    n_rollout, time_step = info['n_rollout'], info['time_step']
    dt, video, args, phase = info['dt'], info['video'], info['args'], info['phase']

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

    attr_dim = args.attr_dim  # attrs: actuated/soft/rigid/fixed

    state_dim = args.state_dim  # x, y, xdot, ydot
    action_dim = args.action_dim
    param_dim = args.param_dim  # n_box, k, damping, init_p

    act_scale = 650.
    act_delta = 200.

    # attr, state, action
    stats = [init_stat(attr_dim), init_stat(state_dim), init_stat(action_dim)]

    if args.fix_param == 1:
        raise AssertionError("Unimplemented")
    else:
        engine = SoftEngine(dt, state_dim, action_dim, param_dim)

    group_size = args.group_size
    sub_dataset_size = n_rollout * args.num_workers // args.n_splits
    print('group size', group_size, 'sub_dataset_size', sub_dataset_size)
    assert n_rollout % group_size == 0
    assert args.n_rollout % args.n_splits == 0

    bar = ProgressBar()
    for i in bar(range(n_rollout)):
        rollout_idx = thread_idx * n_rollout + i
        group_idx = rollout_idx // group_size
        sub_idx = rollout_idx // sub_dataset_size

        num_obj_range = args.num_obj_range if phase in {'train', 'valid'} else args.extra_num_obj_range
        num_obj = num_obj_range[sub_idx]

        rollout_dir = os.path.join(data_dir, str(rollout_idx))
        param_file = os.path.join(data_dir, str(group_idx) + '.param')
        os.system('mkdir -p ' + rollout_dir)

        if rollout_idx % group_size == 0:
            engine.init(param=(num_obj, None, None, None))
            torch.save(engine.get_param(), param_file)
        else:
            while not os.path.isfile(param_file):
                time.sleep(0.5)
            param = torch.load(param_file)
            engine.init(param=param)

        # act_t_param = np.zeros((engine.n_box, 1))

        for j in range(time_step):
            box_type = engine.init_p[:, 2]
            act_t = np.zeros((engine.n_box, action_dim))

            for k in range(engine.n_box):
                if box_type[k] == 0:
                    '''
                    # if this is an actuated box
                    if j == 0:
                        act_t_param[k] = np.array([rand_float(0., 1.)])

                    if act_t_param[k] < 0.5:
                        # using random action
                        act_t[k] = rand_float(-act_scale, act_scale)

                    else:
                    '''
                    # using smooth action
                    if j == 0:
                        act_t[k] = rand_float(-act_delta, act_delta)
                    else:
                        act_t[k] = actions_all[j - 1, k] + rand_float(-act_delta, act_delta)
                        act_t[k] = np.clip(act_t[k], -act_scale, act_scale)

            engine.set_action(act_t)

            states = engine.get_state()
            actions = engine.get_action()

            pos = states[:, :8].copy()
            vec = states[:, 8:].copy()

            '''reset velocity'''
            if j > 0:
                vec = (pos - states_all[j - 1, :, :8]) / dt

            if j == 0:
                attrs_all = np.zeros((time_step, num_obj, attr_dim))
                states_all = np.zeros((time_step, num_obj, state_dim))
                actions_all = np.zeros((time_step, num_obj, action_dim))

            '''attrs: actuated/soft/rigid/fixed'''
            assert attr_dim == 4
            attrs = np.zeros((num_obj, attr_dim))

            for k in range(engine.n_box):
                attrs[k, int(engine.init_p[k, 2])] = 1

            assert np.sum(attrs[:, 0]) == np.sum(engine.init_p[:, 2] == 0)
            assert np.sum(attrs[:, 1]) == np.sum(engine.init_p[:, 2] == 1)
            assert np.sum(attrs[:, 2]) == np.sum(engine.init_p[:, 2] == 2)
            assert np.sum(attrs[:, 3]) == np.sum(engine.init_p[:, 2] == 3)
            assert (np.sum(attrs, 1) == 1).all()

            attrs_all[j] = attrs
            states_all[j, :, :8] = pos
            states_all[j, :, 8:] = vec
            actions_all[j] = actions

            data = [attrs, states_all[j], actions_all[j]]

            store_data(data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))

            engine.step()

        datas = [attrs_all.astype(np.float64), states_all.astype(np.float64), actions_all.astype(np.float64)]

        for j in range(len(stats)):
            stat = init_stat(stats[j].shape[0])
            stat[:, 0] = np.mean(datas[j], axis=(0, 1))[:]
            stat[:, 1] = np.std(datas[j], axis=(0, 1))[:]
            stat[:, 2] = datas[j].shape[0]
            stats[j] = combine_stat(stats[j], stat)

    return stats


def gen_Swim(info):
    thread_idx, data_dir, data_names = info['thread_idx'], info['data_dir'], info['data_names']
    n_rollout, time_step = info['n_rollout'], info['time_step']
    dt, video, args, phase = info['dt'], info['video'], info['args'], info['phase']

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

    attr_dim = args.attr_dim  # actuated, soft, rigid
    state_dim = args.state_dim  # x, y, xdot, ydot
    action_dim = args.action_dim
    param_dim = args.param_dim  # n_box, k, damping, init_p

    act_scale = 500.
    act_delta = 250.

    # attr, state, action
    stats = [init_stat(attr_dim), init_stat(state_dim), init_stat(action_dim)]

    if args.fix_param == 1:
        raise AssertionError("Unimplemented")
    else:
        engine = SwimEngine(dt, state_dim, action_dim, param_dim)

    group_size = args.group_size
    sub_dataset_size = n_rollout * args.num_workers // args.n_splits
    print('group size', group_size, 'sub_dataset_size', sub_dataset_size)
    assert n_rollout % group_size == 0
    assert args.n_rollout % args.n_splits == 0

    bar = ProgressBar()
    for i in bar(range(n_rollout)):
        rollout_idx = thread_idx * n_rollout + i
        group_idx = rollout_idx // group_size
        sub_idx = rollout_idx // sub_dataset_size

        num_obj_range = args.num_obj_range if phase in {'train', 'valid'} else args.extra_num_obj_range
        num_obj = num_obj_range[sub_idx]

        rollout_dir = os.path.join(data_dir, str(rollout_idx))
        param_file = os.path.join(data_dir, str(group_idx) + '.param')
        os.system('mkdir -p ' + rollout_dir)

        if rollout_idx % group_size == 0:
            init_p = None if not args.regular_data else sample_init_p_flight(n_box=num_obj, aug=True, train=phase=='train')
            engine.init(param=(num_obj, None, None, init_p))
            torch.save(engine.get_param(), param_file)
        else:
            while not os.path.isfile(param_file):
                time.sleep(0.5)
            param = torch.load(param_file)
            engine.init(param=param)

        act_t_param = np.zeros((engine.n_box, 3))

        for j in range(time_step):
            box_type = engine.init_p[:, 2]
            act_t = np.zeros((engine.n_box, action_dim))

            for k in range(engine.n_box):
                if box_type[k] == 0:
                    # if this is an actuated box
                    if j == 0:
                        act_t_param[k] = np.array([rand_float(0., 1.), rand_float(1., 2.5), rand_float(0, np.pi * 2)])

                    if act_t_param[k, 0] < 0.3:
                        # using smooth action
                        if j == 0:
                            act_t[k] = rand_float(-act_delta, act_delta)
                        else:
                            lo = max(actions_all[j - 1, k] - act_delta, - act_scale - 20)
                            hi = min(actions_all[j - 1, k] + act_delta, act_scale + 20)
                            act_t[k] = rand_float(lo, hi)
                            act_t[k] = np.clip(act_t[k], -act_scale, act_scale)

                    elif act_t_param[k, 0] < 0.6:
                        # using random action
                        act_t[k] = rand_float(-act_scale, act_scale)

                    else:
                        # using sin action
                        act_t[k] = np.sin(j / act_t_param[k, 1] + act_t_param[k, 2]) * \
                                rand_float(act_scale / 2., act_scale)

            engine.set_action(act_t)

            states = engine.get_state()
            actions = engine.get_action()

            pos = states[:, :8].copy()
            vec = states[:, 8:].copy()

            '''reset velocity'''
            if j > 0:
                vec = (pos - states_all[j - 1, :, :8]) / dt

            if j == 0:
                attrs_all = np.zeros((time_step, num_obj, attr_dim))
                states_all = np.zeros((time_step, num_obj, state_dim))
                actions_all = np.zeros((time_step, num_obj, action_dim))

            '''attrs: actuated/soft/rigid'''
            assert attr_dim == 3
            attrs = np.zeros((num_obj, attr_dim))

            for k in range(engine.n_box):
                attrs[k, int(engine.init_p[k, 2])] = 1

            assert np.sum(attrs[:, 0]) == np.sum(engine.init_p[:, 2] == 0)
            assert np.sum(attrs[:, 1]) == np.sum(engine.init_p[:, 2] == 1)
            assert np.sum(attrs[:, 2]) == np.sum(engine.init_p[:, 2] == 2)

            attrs_all[j] = attrs
            states_all[j, :, :8] = pos
            states_all[j, :, 8:] = vec
            actions_all[j] = actions

            data = [attrs, states_all[j], actions_all[j]]

            store_data(data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))

            engine.step()

        datas = [attrs_all.astype(np.float64), states_all.astype(np.float64), actions_all.astype(np.float64)]

        for j in range(len(stats)):
            stat = init_stat(stats[j].shape[0])
            stat[:, 0] = np.mean(datas[j], axis=(0, 1))[:]
            stat[:, 1] = np.std(datas[j], axis=(0, 1))[:]
            stat[:, 2] = datas[j].shape[0]
            stats[j] = combine_stat(stats[j], stat)

    return stats



class PhysicsDataset(Dataset):

    def __init__(self, args, phase):
        self.args = args
        self.phase = phase
        self.data_dir = os.path.join(self.args.dataf, phase)
        if gethostname().startswith('netmit') and phase == 'extra':
            self.data_dir = self.args.dataf + '_' + phase

        self.stat_path = os.path.join(self.args.dataf, 'stat.h5')
        self.stat = None

        os.system('mkdir -p ' + self.data_dir)

        if args.env in ['Rope', 'Soft', 'Swim']:
            self.data_names = ['attrs', 'states', 'actions']
        else:
            raise AssertionError("Unknown env")

        ratio = self.args.train_valid_ratio
        if phase == 'train':
            self.n_rollout = int(self.args.n_rollout * ratio)
        elif phase in {'valid', 'extra'}:
            self.n_rollout = self.args.n_rollout - int(self.args.n_rollout * ratio)
        else:
            raise AssertionError("Unknown phase")

        self.T = self.args.len_seq

    def load_data(self):
        self.stat = load_data(self.data_names, self.stat_path)

    def gen_data(self):
        # if the data hasn't been generated, generate the data
        n_rollout, time_step, dt = self.n_rollout, self.args.time_step, self.args.dt
        assert n_rollout % self.args.num_workers == 0

        print("Generating data ... n_rollout=%d, time_step=%d" % (n_rollout, time_step))

        infos = []
        for i in range(self.args.num_workers):
            info = {'thread_idx': i,
                    'data_dir': self.data_dir,
                    'data_names': self.data_names,
                    'n_rollout': n_rollout // self.args.num_workers,
                    'time_step': time_step,
                    'dt': dt,
                    'video': False,
                    'phase': self.phase,
                    'args': self.args}

            infos.append(info)

        cores = self.args.num_workers
        pool = mp.Pool(processes=cores)

        env = self.args.env

        if env == 'Rope':
            data = pool.map(gen_Rope, infos)
        elif env == 'Soft':
            data = pool.map(gen_Soft, infos)
        elif env == 'Swim':
            data = pool.map(gen_Swim, infos)
        else:
            raise AssertionError("Unknown env")

        print("Training data generated, warpping up stats ...")

        if self.phase == 'train':
            # states [x, y, angle, xdot, ydot, angledot], action [x, xdot]
            if env in ['Rope', 'Soft', 'Swim']:
                self.stat = [init_stat(self.args.attr_dim),
                             init_stat(self.args.state_dim),
                             init_stat(self.args.action_dim)]

            for i in range(len(data)):
                for j in range(len(self.stat)):
                    self.stat[j] = combine_stat(self.stat[j], data[i][j])

            if self.args.gen_stat:
                print("Storing stat to %s" % self.stat_path)
                store_data(self.data_names, self.stat, self.stat_path)
            else:
                print("stat will be discarded")
        else:
            print("Loading stat from %s ..." % self.stat_path)

            if env in ['Rope', 'Soft', 'Swim']:
                self.stat = load_data(self.data_names, self.stat_path)

    def __len__(self):
        return self.n_rollout * (self.args.time_step - self.T)

    def __getitem__(self, idx):
        idx_rollout = idx // (self.args.time_step - self.T)
        idx_timestep = idx % (self.args.time_step - self.T)

        # prepare input data
        seq_data = None
        for t in range(self.T + 1):
            data_path = os.path.join(self.data_dir, str(idx_rollout), str(idx_timestep + t) + '.h5')
            data = load_data(self.data_names, data_path)
            data = prepare_input(data, self.stat, self.args)
            if seq_data is None:
                seq_data = [[d] for d in data]
            else:
                for i, d in enumerate(data):
                    seq_data[i].append(d)
        seq_data = [np.array(d).astype(np.float32) for d in seq_data]

        return seq_data


if __name__ == '__main__':
    from easydict import EasyDict

    args = EasyDict()
    args.dataf = 'data'

    args.train_valid_ratio = 0.9
    args.num_workers = 10
    args.len_seq = 64

    # args.env = 'Rope'
    args.env = 'Soft'
    args.dataf = 'data/' + args.dataf + '_' + args.env

    if args.env == 'Rope':
        args.dt = 1.0 / 50
        args.n_rollout = 1000
        args.time_step = 101

        args.attr_dim = 2
        args.state_dim = 4
        args.action_dim = 1

        args.relation_dim = 8

        args.param_dim = 5
        args.n_splits = 5

    elif args.env == 'Soft':
        args.dt = 1.0 / 50
        args.n_rollout = 1000
        args.time_step = 101

        args.attr_dim = 3  # actuated, soft tissue, rigid tissue
        args.state_dim = 4
        args.action_dim = 1

        args.relation_dim = 9

        args.param_dim = 4
        args.n_splits = 10

    dataset = PhysicsDataset(args, phase='train')
    dataset.gen_data()
