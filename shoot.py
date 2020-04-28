import os
import numpy as np
import cvxpy as cp
from cvxpy import quad_form

import torch
import torch.optim as optim
import torch.nn.functional as F

from physics_engine import RopeEngine, SoftEngine, SwimEngine
from data import load_data, normalize, denormalize
from models.CompositionalKoopmanOperators import CompositionalKoopmanOperators, regularize_state_Soft
from models.KoopmanBaselineModel import KoopmanBaseline
from utils import to_var, to_np, Tee, norm, get_flat, print_args

from progressbar import ProgressBar

from config import gen_args
from socket import gethostname

args = gen_args()

os.system("mkdir -p " + args.shootf)

log_path = os.path.join(args.shootf, 'log.txt')
tee = Tee(log_path, 'w')

print_args(args)

print(f"Load stored dataset statistics from {args.stat_path}!")
stat = load_data(args.data_names, args.stat_path)

data_names = ['attrs', 'states', 'actions']
prepared_names = ['attrs', 'states', 'actions', 'rel_attrs']
data_dir = os.path.join(args.dataf, args.shoot_set)

if args.shoot_set == 'extra' and gethostname().startswith('netmit'):
    data_dir = args.dataf + '_' + args.shoot_set

'''
model
'''
# build model
use_gpu = torch.cuda.is_available()
if not args.baseline:
    """ Koopman model"""
    model = CompositionalKoopmanOperators(args, residual=False, use_gpu=use_gpu)

    # load pretrained checkpoint
    if args.shoot_epoch == -1:
        model_path = os.path.join(args.outf, 'net_best.pth')
    else:
        model_path = os.path.join(args.outf, 'net_epoch_%d_iter_%d.pth' % (args.shoot_epoch, args.shoot_iter))

    print("Loading saved ckp from %s" % model_path)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:0' if use_gpu else 'cpu')))
    model.eval()
    if use_gpu: model.cuda()

else:
    """ Koopman Baselinese """
    model = KoopmanBaseline(args)

'''
shoot
'''

if args.env == 'Rope':
    engine = RopeEngine(args.dt, args.state_dim, args.action_dim, args.param_dim)
elif args.env == 'Soft':
    engine = SoftEngine(args.dt, args.state_dim, args.action_dim, args.param_dim)
elif args.env == 'Swim':
    engine = SwimEngine(args.dt, args.state_dim, args.action_dim, args.param_dim)
else:
    assert False


def get_more_trajectories(roll_idx):
    group_idx = roll_idx // args.group_size
    offset = group_idx * args.group_size

    all_seq = [[], [], [], []]

    for i in range(1, args.fit_num + 1):
        new_idx = (roll_idx + i - offset) % args.group_size + offset
        seq_data = load_data(prepared_names, os.path.join(data_dir, str(new_idx) + '.rollout.h5'))
        for j in range(4):
            all_seq[j].append(seq_data[j])

    all_seq = [np.array(all_seq[j], dtype=np.float32) for j in range(4)]
    return all_seq

def mpc_qp(g_cur, g_goal, time_cur, T, rel_attrs, A_t, B_t, Q, R, node_attrs=None,
           actions=None, gt_info=None):
    """
    Model Predictive Control + Quadratic Programming
    :param rel_attrs: N x N x relation_dim
    :param node_attrs: N x attributes_dim
    :return action sequence u: T - 1 x N  x action_dim
    """

    n_obj = engine.num_obj
    constraints = []

    if not args.baseline:
        D = args.g_dim
    else:
        D = g_goal.shape[-1]

    if args.fit_type == 'structured':
        dim_a = args.action_dim
        g = cp.Variable((T * n_obj, D))
        u = cp.Variable(((T - 1) * n_obj, args.action_dim))
        augG = cp.Variable(((T - 1) * n_obj, D * args.relation_dim))
        augU = cp.Variable(((T - 1) * n_obj, args.action_dim * args.relation_dim))

        for t in range(T - 1):
            st_idx = t * n_obj
            ed_idx = (t + 1) * n_obj
            for r in range(args.relation_dim):
                constraints.append(augG[st_idx:ed_idx, r * D: (r + 1) * D] ==
                                   rel_attrs[:, :, r] * g[st_idx:ed_idx])
            for r in range(args.relation_dim):
                constraints.append(augU[st_idx:ed_idx, r * dim_a: (r + 1) * dim_a] ==
                                   rel_attrs[:, :, r] * u[st_idx:ed_idx])

        cost = 0

        for idx in range(n_obj):
            # constrain the initial g
            constraints.append(g[idx] == g_cur[idx])

            for t in range(1, T):
                cur_idx = t * n_obj + idx
                prv_idx = (t - 1) * n_obj + idx

                zero_normed = -stat[2][:, 0] / stat[2][:, 1]
                act_scale_max_normed = (args.act_scale - stat[2][:, 0]) / stat[2][:, 1]
                act_scale_min_normed = (- args.act_scale - stat[2][:, 0]) / stat[2][:, 1]
                constraints.append(u[prv_idx] >= act_scale_min_normed)
                constraints.append(u[prv_idx] <= act_scale_max_normed)

                if args.env == 'Rope':
                    if idx == 0:
                        # first mass: action_y = 0 (no action_y now)
                        pass
                    else:
                        # other mass: action_x = action_y = 0
                        constraints.append(u[prv_idx][:] == zero_normed)

                elif args.env in ['Soft', 'Swim']:
                    if node_attrs[idx, 0] < 1e-6:
                        # if there is no actuation
                        constraints.append(u[prv_idx][:] == zero_normed)
                    else:
                        pass

                constraints.append(g[cur_idx] == A_t * augG[prv_idx] + B_t * augU[prv_idx])
                # penalize large actions
                cost += quad_form(u[prv_idx] - zero_normed, R)
            cost += quad_form(g[(T - 1) * n_obj + idx] - g_goal[idx], Q)

    elif args.fit_type == 'unstructured':

        zero_normed = -stat[2][:, 0] / stat[2][:, 1]

        g = cp.Variable((T, n_obj * args.g_dim))
        u = cp.Variable((T - 1, n_obj * args.action_dim))

        cost = 0

        constraints.append(g[0] == g_cur.ravel())

        for t in range(1, T):

            act_scale_normed = (args.act_scale - stat[2][:, 0]) / stat[2][:, 1]
            act_scale_normed = np.repeat(act_scale_normed, n_obj, 0)
            constraints.append(u[t - 1] >= - act_scale_normed)
            constraints.append(u[t - 1] <= act_scale_normed)

            if args.env == 'Rope':
                # set action on balls to zeros expect the first one
                for idx in range(1, n_obj):
                    constraints.append(u[t - 1][idx] == zero_normed)

            elif args.env in ['Soft', 'Swim']:
                for idx in range(0, n_obj):
                    if node_attrs[idx, 0] < 1e-6:
                        constraints.append(u[t - 1][idx * args.action_dim: (idx + 1) * args.action_dim] == zero_normed)

            constraints.append(g[t] == A_t * g[t - 1] + B_t * u[t - 1])

            for i in range(n_obj):
                cost += quad_form(u[t - 1][i * args.action_dim:(i + 1) * args.action_dim] - zero_normed, R)

        for i in range(n_obj):
            cost += quad_form(g[T - 1][i * args.g_dim:(i + 1) * args.g_dim] - g_goal[i], Q)

    elif args.fit_type == 'diagonal':

        zero_normed = -stat[2][:, 0] / stat[2][:, 1]

        g = cp.Variable((T, n_obj * args.g_dim))
        u = cp.Variable((T - 1, n_obj * args.action_dim))

        cost = 0
        constraints.append(g[0] == g_cur.ravel())

        for t in range(1, T):
            act_scale_normed = (args.act_scale - stat[2][:, 0]) / stat[2][:, 1]
            act_scale_normed = np.repeat(act_scale_normed, n_obj, 0)
            constraints.append(u[t - 1] >= - act_scale_normed)
            constraints.append(u[t - 1] <= act_scale_normed)
            if args.env == 'Rope':
                # set action on balls to zeros expect the first one
                for idx in range(1, n_obj):
                    constraints.append(u[t - 1][idx] == zero_normed)
            elif args.env in ['Soft', 'Swim']:
                for idx in range(0, n_obj):
                    if node_attrs[idx, 0] < 1e-6:
                        constraints.append(u[t - 1][idx * args.action_dim: (idx + 1) * args.action_dim] == zero_normed)

            for i in range(n_obj):
                t1 = A_t * g[t - 1][i * args.g_dim:(i + 1) * args.g_dim]
                t2 = B_t * u[t - 1][i * args.action_dim:(i + 1) * args.action_dim]
                if args.env == 'Rope':
                    t2 = t2[:, 0]
                constraints.append(g[t][i * args.g_dim:(i + 1) * args.g_dim] == t1 + t2)
                cost += quad_form(u[t - 1][i * args.action_dim:(i + 1) * args.action_dim] - zero_normed, R)
        for i in range(n_obj):
            cost += quad_form(g[T - 1][i * args.g_dim:(i + 1) * args.g_dim] - g_goal[i], Q)

    objective = cp.Minimize(cost)
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    u_val = u.value
    g_val = g.value
    u = u_val.reshape(T - 1, n_obj, args.action_dim)

    u = denormalize([u], [stat[2]])[0]
    g = g_val.reshape(T, n_obj, D)

    return u


def shoot_mpc_qp(roll_idx):
    print(f'\n=== Model Based Control on Example {roll_idx} ===')

    '''
    load data
    '''
    seq_data = load_data(prepared_names, os.path.join(data_dir, str(roll_idx) + '.rollout.h5'))
    attrs, states, actions, rel_attrs = [to_var(d.copy(), use_gpu=use_gpu) for d in seq_data]

    seq_data = denormalize(seq_data, stat)
    attrs_gt, states_gt, actions_gt = seq_data[:3]

    '''
    setup engine
    '''
    param_file = os.path.join(data_dir, str(roll_idx // args.group_size) + '.param')
    param = torch.load(param_file)
    engine.init(param)
    n_obj = engine.num_obj

    '''
    fit koopman
    '''
    print('===> system identification!')
    fit_data = get_more_trajectories(roll_idx)
    fit_data = [to_var(d, use_gpu=use_gpu) for d in fit_data]
    bs = args.fit_num

    attrs_flat = get_flat(fit_data[0])
    states_flat = get_flat(fit_data[1])
    actions_flat = get_flat(fit_data[2])
    rel_attrs_flat = get_flat(fit_data[3])

    g = model.to_g(attrs_flat, states_flat, rel_attrs_flat, args.pstep)
    g = g.view(torch.Size([bs, args.time_step]) + g.size()[1:])

    G_tilde = g[:, :-1]
    H_tilde = g[:, 1:]
    U_left = fit_data[2][:, :-1]

    G_tilde = get_flat(G_tilde, keep_dim=True)
    H_tilde = get_flat(H_tilde, keep_dim=True)
    U_left = get_flat(U_left, keep_dim=True)

    A, B, fit_err = model.system_identify(G=G_tilde, H=H_tilde, U=U_left,
                                          rel_attrs=fit_data[3][:1, 0], I_factor=args.I_factor)

    '''
    shooting
    '''
    print('===> model based control start!')
    # current can not set engine to a middle state
    assert args.roll_start == 0

    start_step = args.roll_start
    g_start_v = model.to_g(attrs=attrs[start_step:start_step + 1], states=states[start_step:start_step + 1],
                           rel_attrs=rel_attrs[start_step:start_step + 1], pstep=args.pstep)
    g_start = to_np(g_start_v[0])

    if args.env == 'Rope':
        goal_step = args.roll_step + args.roll_start
    elif args.env == 'Soft':
        goal_step = args.roll_step + args.roll_start
    elif args.env == 'Swim':
        goal_step = args.roll_step + args.roll_start

    g_goal_v = model.to_g(attrs=attrs[goal_step:goal_step + 1], states=states[goal_step:goal_step + 1],
                          rel_attrs=rel_attrs[goal_step:goal_step + 1], pstep=args.pstep)
    g_goal = to_np(g_goal_v[0])

    states_start = states_gt[start_step]
    states_goal = states_gt[goal_step]
    states_roll = np.zeros((args.roll_step + 1, n_obj, args.state_dim))
    states_roll[0] = states_start

    control = np.zeros((args.roll_step + 1, n_obj, args.action_dim))
    # control_v = to_var(control, use_gpu, requires_grad=True)
    bar = ProgressBar()
    for step in bar(range(args.roll_step)):
        states_input = normalize([states_roll[step:step + 1]], [stat[1]])[0]
        states_input_v = to_var(states_input, use_gpu=use_gpu)
        g_cur_v = model.to_g(attrs=attrs[:1], states=states_input_v,
                             rel_attrs=rel_attrs[:1], pstep=args.pstep)
        g_cur = to_np(g_cur_v[0])

        '''
        setup parameters
        '''
        T = args.roll_step - step + 1

        A_v, B_v = model.A, model.B
        A_t = to_np(A_v[0]).T
        B_t = to_np(B_v[0]).T

        if not args.baseline:
            Q = np.eye(args.g_dim)
        else:
            Q = np.eye(g_goal.shape[-1])

        if args.env == 'Rope':
            R_factor = 0.01
        elif args.env == 'Soft':
            R_factor = 0.001
        elif args.env == 'Swim':
            R_factor = 0.0001
        else:
            assert False

        R = np.eye(args.action_dim) * R_factor

        '''
        generate action
        '''
        rel_attrs_np = to_np(rel_attrs)[0]
        assert args.optim_type == 'qp'
        if step % args.feedback == 0:
            node_attrs = attrs_gt[0] if args.env in ['Soft', 'Swim'] else None
            u = mpc_qp(g_cur, g_goal, step, T, rel_attrs_np, A_t, B_t, Q, R, node_attrs=node_attrs,
                       actions=to_np(actions[step:]),
                       gt_info=[param, states_gt[goal_step:goal_step + 1], attrs[step:step + T],
                                rel_attrs[step:step + T]])
        else:
            u = u[1:]
            pass

        '''
        execute action
        '''
        engine.set_action(u[0])  # execute the first action
        control[step] = engine.get_action()
        engine.step()
        states_roll[step + 1] = engine.get_state()

    '''
    render
    '''
    engine.render(states_roll, control, param, act_scale=args.act_scale, video=True, image=False,
                  path=os.path.join(args.shootf, str(roll_idx) + '.shoot'),
                  states_gt=np.tile(states_gt[goal_step:goal_step + 1], (args.roll_step + 1, 1, 1)),
                  count_down=True, gt_border=True)

    states_result = states_roll[args.roll_step]

    states_goal_normalized = normalize([states_goal], [stat[1]])[0]
    states_result_normalized = normalize([states_result], [stat[1]])[0]

    return norm(states_goal - states_result), (states_goal, states_result, states_goal_normalized, states_result_normalized)


if __name__ == '__main__':
    os.system('mkdir -p ' + args.shootf)
    num_train = int(args.n_rollout * args.train_valid_ratio)
    num_valid = args.n_rollout - num_train
    ls_rollout_idx = np.arange(0, num_valid, num_valid // args.group_size // 5)

    if args.demo:
        ls_rollout_idx = np.arange(8) * 25

    for roll_idx in ls_rollout_idx:
        shoot_mpc_qp(roll_idx)
