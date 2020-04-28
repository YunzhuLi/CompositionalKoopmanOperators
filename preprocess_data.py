from config import gen_args
import os
from utils import *
from data import prepare_input
from progressbar import ProgressBar
import multiprocessing as mp
from socket import gethostname

args = gen_args()

data_names = ['attrs', 'states', 'actions']
prepared_names = ['attrs', 'states', 'actions', 'rel_attrs']

stat_path = os.path.join(args.dataf, 'stat.h5')
stat = load_data(data_names, stat_path)


def prepare_seq(info):
    phase, rollout_idx = info
    data_dir = os.path.join(args.dataf, phase)
    if phase == 'extra' and gethostname().startswith('netmit'):
        data_dir = args.dataf + '_' + phase

    # get param
    if args.env == 'Rope':
        param = None
    elif args.env in ['Soft', 'Swim']:
        param_file = os.path.join(data_dir, str(rollout_idx // args.group_size) + '.param')
        param = torch.load(param_file)
    else:
        assert False

    # prepare input data
    seq_data = None
    for t in range(args.time_step):
        data_path = os.path.join(data_dir, str(rollout_idx), str(t) + '.h5')
        data = load_data(data_names, data_path)
        data = prepare_input(data, stat, args, param=param)
        if seq_data is None:
            seq_data = [[d] for d in data]
        else:
            for i, d in enumerate(data):
                seq_data[i].append(d)
    seq_data = [np.array(d).astype(np.float32) for d in seq_data]

    assert len(seq_data) == len(prepared_names)

    store_data(prepared_names, seq_data, os.path.join(data_dir, str(rollout_idx) + '.rollout.h5'))


def sub_thread(info):
    n_workers, idx, n_rollout, phase = info
    bar = ProgressBar()
    n = n_rollout // n_workers
    for i in bar(range(n)):
        prepare_seq(info=(phase, n * idx + i))


n_workers = 10
pool = mp.Pool(processes=n_workers)

num_train = int(args.n_rollout * args.train_valid_ratio)
num_valid = args.n_rollout - num_train

infos = [(n_workers, idx, num_train, 'train') for idx in range(n_workers)]
pool.map(sub_thread, infos)

infos = [(n_workers, idx, num_valid, 'valid') for idx in range(n_workers)]
pool.map(sub_thread, infos)