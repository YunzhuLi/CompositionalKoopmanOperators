import argparse

parser = argparse.ArgumentParser()
'''
General
'''
parser.add_argument('--env', default='', required=True, help='Rope | Soft | Swim')
parser.add_argument('--dt', type=float, default=1. / 50.)

'''
Compositional Koopman Operator model
'''
parser.add_argument('--pstep', type=int, default=2, help='number of propagation steps in GNN model')
parser.add_argument('--nf_relation', type=int, default=120, help='length of relation encoding')
parser.add_argument('--nf_particle', type=int, default=100, help='length of object encoding')
parser.add_argument('--nf_effect', type=int, default=100, help='length of effect encoding')
parser.add_argument('--g_dim', type=int, default=32, help='dimention of latent linear dynamics')
parser.add_argument('--fit_type', default='structured',
                    help="what is the structure of AB matrix in koopman: structured | unstructured | diagonal")
# input dimensions
parser.add_argument('--attr_dim', type=int, default=0)
parser.add_argument('--state_dim', type=int, default=0)
parser.add_argument('--action_dim', type=int, default=0)
parser.add_argument('--relation_dim', type=int, default=0)

'''
Koopman baseline with polynomial Koopman basis
'''
parser.add_argument('--baseline', default=False, action='store_true')
parser.add_argument('--baseline_order', type=int, default=3, help='order of polynomial basis')

'''
data
'''
parser.add_argument('--dataf', default='data')
parser.add_argument('--regular_data', type=int, default=0, help='generate regular shape of soft robot (used in Swim env)')
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--gen_data', type=int, default=0, help="whether to generate new data")
parser.add_argument('--gen_stat', type=int, default=1, help="whether to generate statistic for the data")
parser.add_argument('--group_size', type=int, default=25, help='# of episodes sharing the same physical parameters')

'''
train
'''
parser.add_argument('--outf', default='train')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--grad_clip', type=float, default=5.0)
parser.add_argument('--n_epoch', type=int, default=1000)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--log_per_iter', type=int, default=100, help="print log every x iterations")
parser.add_argument('--ckp_per_iter', type=int, default=1000, help="save checkpoint every x iterations")
parser.add_argument('--resume_epoch', type=int, default=-1, help="resume epoch of previous trained checkpoint")
parser.add_argument('--resume_iter', type=int, default=-1, help="resume iteration of previous trained checkpoint")
parser.add_argument('--lambda_loss_metric', type=float, default=0.3)
parser.add_argument('--len_seq', type=int, default=64, help='length of every episodes used in training')

'''
system identification
'''
parser.add_argument('--I_factor', type=float, default=10, help='l2 regularization factor of least-square fitting')
parser.add_argument('--fit_num', type=int, default=8, help='number of episodes used for system identification')

'''
eval
'''
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--evalf', default='eval')
parser.add_argument('--eval_type', default='koopman', help='rollout|valid|koopman')
parser.add_argument('--eval_epoch', type=int, default=-1)
parser.add_argument('--eval_iter', type=int, default=-1)
parser.add_argument('--eval_set', default='valid', help='train|valid|demo')

'''
shoot
'''
parser.add_argument('--shootf', default='shoot')
parser.add_argument('--optim_iter_init', type=int, default=100)
parser.add_argument('--optim_iter', type=int, default=10)
parser.add_argument('--optim_type', default='qp', help="qp|lqr")
parser.add_argument('--feedback', type=int, default=1, help="optimize the control signals every x steps")
parser.add_argument('--shoot_set', default='valid', help='train|valid|demo')
parser.add_argument('--roll_start', type=int, default=0)
parser.add_argument('--roll_step', type=int, default=40)
parser.add_argument('--shoot_epoch', type=int, default=-1)
parser.add_argument('--shoot_iter', type=int, default=-1)




def gen_args():
    args = parser.parse_args()
    assert args.batch_size == args.fit_num
    if args.env == 'Rope':
        args.data_names = ['attrs', 'states', 'actions']

        args.n_rollout = 10000
        args.train_valid_ratio = 0.9

        args.time_step = 101
        # one hot to indicate root/children
        args.attr_dim = 2
        # state [x, y, xdot, ydot]
        args.state_dim = 4
        # action [x]
        args.action_dim = 1
        # relation [spring, ghost spring]
        args.relation_dim = 8

        args.param_dim = 5

        args.n_splits = 5
        args.num_obj_range = [*range(5, 5 + 5)]
        args.extra_num_obj_range = [10, 11, 12, 13, 14]

        args.act_scale = 2.

    elif args.env == 'Soft':
        args.data_names = ['attrs', 'states', 'actions']

        args.n_rollout = 50000
        args.train_valid_ratio = 0.9

        args.time_step = 101
        # one hot to indicate actuated / soft / rigid / fixed
        args.attr_dim = 4
        # state [x, y] * 4 + [xdot, ydot] * 4
        args.state_dim = 16
        # action 1-dim scalar of extending or contracting
        args.action_dim = 1
        # relation: #relations types = #spaical position types * #box types
        args.relation_dim = 9 * 4

        args.param_dim = 4
        args.n_splits = 5
        args.num_obj_range = [*range(5, 5 + 5)]
        args.extra_num_obj_range = [10, 11, 12, 13, 14]

        args.act_scale = 650.

    elif args.env == 'Swim':
        args.data_names = ['attrs', 'states', 'actions']

        args.n_rollout = 50000
        args.train_valid_ratio = 0.9

        args.time_step = 101
        # one hot to indicate actuated / soft / rigid
        args.attr_dim = 3
        # state [x, y] * 4 + [xdot, ydot] * 4
        args.state_dim = 16
        # action 1-dim scalar of extending or contracting
        args.action_dim = 1
        # relation: #relations types = #spaical position types * #box types
        args.relation_dim = 9 * 3

        args.param_dim = 4
        args.n_splits = 5
        args.num_obj_range = [*range(5, 5 + 5)]
        args.extra_num_obj_range = [10, 11, 12, 13, 14]

        args.act_scale = 500.

    else:
        raise AssertionError("Unsupported env")

    assert args.n_rollout % (args.group_size * args.n_splits * args.batch_size) == 0

    args.demo = args.eval_set == 'demo' or args.shoot_set == 'demo'
    data_root = 'data'
    args.dataf = data_root + '/' + args.dataf + '_' + args.env

    dump_prefix = 'dump_{}/'.format(args.env)
    args.outf = dump_prefix + args.outf
    args.evalf = dump_prefix + args.evalf
    args.shootf = dump_prefix + args.shootf
    args.tmpf = dump_prefix + 'tmp'
    args.outf = args.outf + '_' + args.env
    args.stat_path = args.dataf + '/' + ('stat.h5' if not args.demo else 'stat_demo.h5')

    if not args.baseline:
        # compositional koopman operators
        args.outf += '_CKO_pstep_' + str(args.pstep)
        args.outf += '_lenseq_' + str(args.len_seq)
        args.outf += '_gdim_' + str(args.g_dim)
        args.outf += '_bs_' + str(args.batch_size)
        args.outf += '_' + str(args.fit_type)

        args.evalf += '_CKO_pstep_' + str(args.pstep)
        args.evalf += '_lenseq_' + str(args.len_seq)
        args.evalf += '_gdim_' + str(args.g_dim)
        args.evalf += '_fitnum_' + str(args.fit_num)
        args.evalf += '_' + str(args.fit_type)
        args.evalf += '_' + str(args.eval_set)
        if args.eval_epoch > -1:
            args.evalf += '_epoch_' + str(args.eval_epoch)
            args.evalf += '_iter_' + str(args.eval_iter)
        else:
            args.evalf += '_epoch_best'

        args.shootf += '_CKO_pstep_' + str(args.pstep)
        args.shootf += '_lenseq_' + str(args.len_seq)
        args.shootf += '_gdim_' + str(args.g_dim)
        args.shootf += '_fitnum_' + str(args.fit_num)
        args.shootf += '_' + args.fit_type
        args.shootf += '_' + args.optim_type
        args.shootf += '_roll_' + str(args.roll_step)
        if args.shoot_epoch > -1:
            args.shootf += '_epoch_' + str(args.shoot_epoch)
            args.shootf += '_iter_' + str(args.shoot_iter)
        else:
            args.shootf += '_epoch_best'

        args.shootf += '_feedback_' + str(args.feedback)
        args.shootf += '_' + str(args.shoot_set)

        # for demo
        if args.demo:
            args.outf = dump_prefix + f'train_{args.env}_CKO_demo'
            args.evalf = dump_prefix + f'eval_{args.env}_CKO_demo'
            args.shootf = dump_prefix + f'shoot_{args.env}_CKO_demo'

    else:

        args.evalf += '_KoopmanBaseline'
        args.evalf += '_fitnum_' + str(args.fit_num)
        args.evalf += '_' + str(args.fit_type)
        args.evalf += '_I_' + str(args.I_factor)
        args.evalf += '_order_' + str(args.baseline_order)
        args.evalf += '_' + str(args.eval_set)

        args.shootf += '_KoopmanBaseline'
        args.shootf += '_fitnum_' + str(args.fit_num)
        args.shootf += '_' + args.fit_type
        args.shootf += '_I_' + str(args.I_factor)
        args.shootf += '_order_' + str(args.baseline_order)
        args.shootf += '_roll_' + str(args.roll_step)
        args.shootf += '_feedback_' + str(args.feedback)

        # for demo
        if args.demo:
            args.outf = dump_prefix + f'train_{args.env}_KoopmanBaseline_demo'
            args.evalf = dump_prefix + f'eval_{args.env}_KoopmanBaseline_demo'
            args.shootf = dump_prefix + f'shoot_{args.env}_KoopmanBaseline_demo'

    return args
