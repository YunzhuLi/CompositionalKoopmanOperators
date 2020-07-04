import os

from config import gen_args
from data import normalize, denormalize
from models.CompositionalKoopmanOperators import CompositionalKoopmanOperators
from models.KoopmanBaselineModel import KoopmanBaseline
from physics_engine import SoftEngine, RopeEngine, SwimEngine
from utils import *
from utils import to_var, to_np, Tee
from progressbar import ProgressBar
import time

args = gen_args()
print_args(args)
'''
args.fit_num is # of trajectories used for SysID
'''
assert args.group_size - 1 >= args.fit_num

data_names = ['attrs', 'states', 'actions']
prepared_names = ['attrs', 'states', 'actions', 'rel_attrs']

data_dir = os.path.join(args.dataf, args.eval_set)

print(f"Load stored dataset statistics from {args.stat_path}!")
stat = load_data(data_names, args.stat_path)

if args.env == 'Rope':
    engine = RopeEngine(args.dt, args.state_dim, args.action_dim, args.param_dim)
elif args.env == 'Soft':
    engine = SoftEngine(args.dt, args.state_dim, args.action_dim, args.param_dim)
elif args.env == 'Swim':
    engine = SwimEngine(args.dt, args.state_dim, args.action_dim, args.param_dim)
else:
    assert False


os.system('mkdir -p ' + args.evalf)
log_path = os.path.join(args.evalf, 'log.txt')
tee = Tee(log_path, 'w')

'''
model
'''
# build model
use_gpu = torch.cuda.is_available()
if not args.baseline:
    """ Koopman model"""
    model = CompositionalKoopmanOperators(args, residual=False, use_gpu=use_gpu)

    # load pretrained checkpoint
    if args.eval_epoch == -1:
        model_path = os.path.join(args.outf, 'net_best.pth')
    else:
        model_path = os.path.join(args.outf, 'net_epoch_%d_iter_%d.pth' % (args.eval_epoch, args.eval_iter))
    print("Loading saved checkpoint from %s" % model_path)
    device = torch.device('cuda:0') if use_gpu else torch.device('cpu')
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()
    if use_gpu: model.cuda()

else:
    """ Koopman Baselinese """
    model = KoopmanBaseline(args)

'''
eval
'''


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

def eval(idx_rollout, video=True):
    print(f'\n=== Forward Simulation on Example {roll_idx} ===')

    seq_data = load_data(prepared_names, os.path.join(data_dir, str(idx_rollout) + '.rollout.h5'))
    attrs, states, actions, rel_attrs = [to_var(d.copy(), use_gpu=use_gpu) for d in seq_data]

    seq_data = denormalize(seq_data, stat)
    attrs_gt, states_gt, action_gt = seq_data[:3]

    param_file = os.path.join(data_dir, str(idx_rollout // args.group_size) + '.param')
    param = torch.load(param_file)
    engine.init(param)

    '''
    fit data
    '''
    fit_data = get_more_trajectories(roll_idx)
    fit_data = [to_var(d, use_gpu=use_gpu) for d in fit_data]
    bs = args.fit_num

    ''' T x N x D (denormalized)'''
    states_pred = states_gt.copy()
    states_pred[1:] = 0

    ''' T x N x D (normalized)'''
    s_pred = states.clone()

    '''
    reconstruct loss
    '''
    attrs_flat = get_flat(fit_data[0])
    states_flat = get_flat(fit_data[1])
    actions_flat = get_flat(fit_data[2])
    rel_attrs_flat = get_flat(fit_data[3])

    g = model.to_g(attrs_flat, states_flat, rel_attrs_flat, args.pstep)
    g = g.view(torch.Size([bs, args.time_step]) + g.size()[1:])

    G_tilde = g[:, :-1]
    H_tilde = g[:, 1:]
    U_tilde = fit_data[2][:, :-1]

    G_tilde = get_flat(G_tilde, keep_dim=True)
    H_tilde = get_flat(H_tilde, keep_dim=True)
    U_tilde = get_flat(U_tilde, keep_dim=True)

    _t = time.time()
    A, B, fit_err = model.system_identify(
        G=G_tilde, H=H_tilde, U=U_tilde, rel_attrs=fit_data[3][:1, 0], I_factor=args.I_factor)
    _t = time.time() - _t

    '''
    predict
    '''

    g = model.to_g(attrs, states, rel_attrs, args.pstep)

    pred_g = None
    for step in range(0, args.time_step - 1):
        # prepare input data

        if step == 0:
            current_s = states[step:step + 1]
            current_g = g[step:step + 1]
            states_pred[step] = states_gt[step]
        else:
            '''current state'''
            if args.eval_type == 'valid':
                current_s = states[step:step + 1]
            elif args.eval_type == 'rollout':
                current_s = s_pred[step:step + 1]

            '''current g'''
            if args.eval_type in {'valid', 'rollout'}:
                current_g = model.to_g(attrs[step:step + 1], current_s, rel_attrs[step:step + 1], args.pstep)
            elif args.eval_type == 'koopman':
                current_g = pred_g

        '''next g'''
        pred_g = model.step(g=current_g, u=actions[step:step + 1], rel_attrs=rel_attrs[step:step + 1])

        '''decode s'''
        pred_s = model.to_s(attrs=attrs[step:step + 1], gcodes=pred_g,
                            rel_attrs=rel_attrs[step:step + 1], pstep=args.pstep)

        pred_s_np_denorm = denormalize([to_np(pred_s)], [stat[1]])[0]

        states_pred[step + 1:step + 2] = pred_s_np_denorm
        d = args.state_dim // 2
        states_pred[step + 1:step + 2, :, :d] = states_pred[step:step + 1, :, :d] + \
                                                args.dt * states_pred[step + 1:step + 2, :, d:]

        s_pred_next = normalize([states_pred[step + 1:step + 2]], [stat[1]])[0]
        s_pred[step + 1:step + 2] = to_var(s_pred_next, use_gpu=use_gpu)

    if video:
        engine.render(states_pred, seq_data[2], param, act_scale=args.act_scale, video=True, image=True,
                      path=os.path.join(args.evalf, str(idx_rollout) + '.pred'),
                      states_gt=states_gt)

if __name__ == '__main__':

    num_train = int(args.n_rollout * args.train_valid_ratio)
    num_valid = args.n_rollout - num_train

    ls_rollout_idx = np.arange(0, num_valid, num_valid // args.n_splits)

    if args.demo:
        ls_rollout_idx = np.arange(8) * 25

    for roll_idx in ls_rollout_idx:
        eval(roll_idx)
