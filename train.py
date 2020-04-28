import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from progressbar import ProgressBar
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from config import gen_args
from data import PhysicsDataset
from data import load_data
from models.CompositionalKoopmanOperators import CompositionalKoopmanOperators
from utils import count_parameters, Tee, AverageMeter, rand_int, mix_iters, get_flat, print_args

args = gen_args()

os.system('mkdir -p ' + args.outf)
os.system('mkdir -p ' + args.dataf)
tee = Tee(os.path.join(args.outf, 'train.log'), 'w')
print_args(args)

# generate data
datasets = {phase: PhysicsDataset(args, phase) for phase in ['train', 'valid']}
for phase in ['train', 'valid']:
    if args.gen_data:
        datasets[phase].gen_data()
    else:
        datasets[phase].load_data()

if args.gen_data:
    print("Preprocessing data ...")
    os.system('python preprocess_data.py --env ' + args.env)

args.stat = datasets['train'].stat


class ShuffledDataset(Dataset):
    def __init__(self,
                 mother_dataset,
                 idx,
                 batch_size):
        self.samples_per_rollout = args.time_step - args.len_seq
        self.mother = mother_dataset
        self.n_rollout = mother_dataset.n_rollout // args.n_splits
        self.idx = idx
        self.prepared_names = ['attrs', 'states', 'actions', 'rel_attrs']
        self.batch_size = batch_size

        self.build_table()

    def __len__(self):
        return self.n_rollout * self.samples_per_rollout

    def build_table(self):
        assert self.n_rollout % args.group_size == 0
        bs = self.batch_size
        num_groups = self.n_rollout // args.group_size

        sample_list = [[] for _ in range(num_groups)]
        for i in range(self.n_rollout):
            for j in range(self.samples_per_rollout):
                gidx = i // args.group_size
                sample_list[gidx].append((i, j))

        '''shuffle sample list'''
        for i in range(num_groups):
            l = sample_list[i]
            random.shuffle(l)

        '''padding samples in the same group such that the size can be divied by the batch size'''
        for i in range(num_groups):
            if len(sample_list[i]) % bs > 0:
                sample_list[i] += sample_list[i][:bs - len(sample_list[i]) % bs]

        '''create batches'''
        batch_list = []
        for i in range(num_groups):
            l = sample_list[i]
            for j in range(len(l) // bs):
                batch_list.append(l[j * bs:j * bs + bs])

        '''merge the batch list to a total sample list'''
        random.shuffle(batch_list)
        total_list = []
        for batch in batch_list:
            total_list += batch
        self.sample_table = total_list

    def __getitem__(self, idx):
        # print('dataset', self.idx, 'sample', idx)
        idx_rollout = self.sample_table[idx][0] + self.n_rollout * self.idx
        idx_timestep = self.sample_table[idx][1]

        # prepare input data
        seq_data = load_data(self.prepared_names, os.path.join(self.mother.data_dir, str(idx_rollout) + '.rollout.h5'))
        seq_data = [d[idx_timestep:idx_timestep + args.len_seq + 1] for d in seq_data]

        # prepare fit data
        fit_idx = rand_int(0, args.group_size - 1)  # new traj idx in group
        fit_idx = fit_idx + idx_rollout // args.group_size * args.group_size  # new traj idx in global
        fit_data = load_data(self.prepared_names, os.path.join(self.mother.data_dir, str(fit_idx) + '.rollout.h5'))

        return seq_data, fit_data


class SubPreparedDataset(Dataset):

    def __init__(self,
                 mother_dataset,
                 idx, ):
        self.samples_per_rollout = args.time_step - args.len_seq
        self.mother = mother_dataset
        self.n_rollout = mother_dataset.n_rollout // args.n_splits
        self.idx = idx
        self.prepared_names = ['attrs', 'states', 'actions', 'rel_attrs']

    def __len__(self):
        return self.n_rollout * self.samples_per_rollout

    def __getitem__(self, idx):
        idx_rollout = idx // self.samples_per_rollout + self.n_rollout * self.idx
        idx_timestep = idx % self.samples_per_rollout

        # prepare input data
        seq_data = load_data(self.prepared_names, os.path.join(self.mother.data_dir, str(idx_rollout) + '.rollout.h5'))
        seq_data = [d[idx_timestep:idx_timestep + args.len_seq + 1] for d in seq_data]

        # prepare fit data
        fit_idx = rand_int(0, args.group_size - 1)  # new traj idx in group
        fit_idx = fit_idx + idx_rollout // args.group_size * args.group_size  # new traj idx in global
        fit_data = load_data(self.prepared_names, os.path.join(self.mother.data_dir, str(fit_idx) + '.rollout.h5'))

        return seq_data, fit_data


def split_dataset(ds):
    assert ds.n_rollout % args.group_size == 0
    assert ds.n_rollout % args.n_splits == 0
    sub_datasets = [ShuffledDataset(mother_dataset=ds, idx=i, batch_size=args.batch_size) for i in range(args.n_splits)]
    return sub_datasets


use_gpu = torch.cuda.is_available()

"""
various number of objects, need mixing datasets
"""

dataloaders = {}
data_n_batches = {}
loaders = {}
for phase in ['train', 'valid']:
    loaders[phase] = [DataLoader(
        dataset=dataset, batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers, )
        for dataset in split_dataset(datasets[phase])]

    dataloaders[phase] = lambda: mix_iters(iters=[iter(loader) for loader in loaders[phase]])

    num_batches = sum(len(loader) for loader in loaders[phase])
    data_n_batches[phase] = num_batches

# Compositional Koopman Operator
model = CompositionalKoopmanOperators(args, residual=False, use_gpu=use_gpu)

# print model #params
print("model #params: %d" % count_parameters(model))

# if resume from a pretrained checkpoint
if args.resume_epoch >= 0:
    model_path = os.path.join(args.outf, 'net_epoch_%d_iter_%d.pth' % (args.resume_epoch, args.resume_iter))
    print("Loading saved ckp from %s" % model_path)
    model.load_state_dict(torch.load(model_path))

# criterion
criterionMSE = nn.MSELoss()

# optimizer
params = model.parameters()
optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, 0.999))
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=2, verbose=True)

if use_gpu:
    model = model.cuda()
    criterionMSE = criterionMSE.cuda()

st_epoch = args.resume_epoch if args.resume_epoch > 0 else 0
best_valid_loss = np.inf

log_fout = open(os.path.join(args.outf, 'log_st_epoch_%d.txt' % st_epoch), 'w')

for epoch in range(st_epoch, args.n_epoch):

    phases = ['train', 'valid'] if args.eval == 0 else ['valid']

    for phase in phases:
        model.train(phase == 'train')
        meter_loss = AverageMeter()
        meter_loss_metric = AverageMeter()
        meter_loss_ae = AverageMeter()
        meter_loss_pred = AverageMeter()
        meter_fit_error = AverageMeter()
        meter_dist_g = AverageMeter()
        meter_dist_s = AverageMeter()

        bar = ProgressBar(max_value=data_n_batches[phase])

        loader = dataloaders[phase]()

        for i, (seq_data, fit_data) in bar(enumerate(loader)):

            attrs, states, actions, rel_attrs = seq_data
            attrs_2, states_2, actions_2, rel_attrs_2 = fit_data
            # print('attrs', attrs.shape)           bs x len_seq x num_obj x attr_dim
            # print('states', states.shape)         bs x len_seq x num_obj x state_dim
            # print('actions', actions.shape)       bs x len_seq x num_obj x action_dim
            # print('rel_attrs', rel_attrs.shape)   bs x len_seq x num_obj x num_obj x rel_dim

            if use_gpu:
                attrs_2, states_2, actions_2, rel_attrs_2 = [x.cuda() for x in fit_data]
            fit_data = [attrs_2, states_2, actions_2, rel_attrs_2]

            with torch.set_grad_enabled(phase == 'train'):
                if use_gpu:
                    attrs, states, actions, rel_attrs = [x.cuda() for x in seq_data]
                data = [attrs, states, actions, rel_attrs]

                T = args.len_seq
                bs = len(attrs)

                """
                flatten fit data
                """
                attrs_flat = get_flat(attrs_2)
                states_flat = get_flat(states_2)
                actions_flat = get_flat(actions_2)
                rel_attrs_flat = get_flat(rel_attrs_2)

                g = model.to_g(attrs_flat, states_flat, rel_attrs_flat, args.pstep)
                g = g.view(torch.Size([bs, args.time_step]) + g.size()[1:])

                """
                fit A with fit data
                !!! need to force that rel_attrs in one group to be the same !!!
                """
                G_tilde = g[:, :-1]
                H_tilde = g[:, 1:]
                U_left = actions_2[:, :-1]

                G_tilde = get_flat(G_tilde, keep_dim=True)
                H_tilde = get_flat(H_tilde, keep_dim=True)
                U_left = get_flat(U_left, keep_dim=True)

                A, B, fit_err = model.system_identify(G=G_tilde, H=H_tilde, U=U_left,
                                                      rel_attrs=rel_attrs[:1, 0], I_factor=args.I_factor)

                model.A = model.A.repeat(bs, 1, 1)
                model.B = model.B.repeat(bs, 1, 1)

                meter_fit_error.update(fit_err.item(), bs)

                """
                forward on sequential data
                """

                attrs_flat = get_flat(attrs)
                states_flat = get_flat(states)
                actions_flat = get_flat(actions)
                rel_attrs_flat = get_flat(rel_attrs)

                g = model.to_g(attrs_flat, states_flat, rel_attrs_flat, args.pstep)

                permu = np.random.permutation(bs * (T + 1))
                split_0 = permu[:bs * (T + 1) // 2]
                split_1 = permu[bs * (T + 1) // 2:]

                dist_g = torch.mean((g[split_0] - g[split_1]) ** 2, dim=(1, 2))
                dist_s = torch.mean((states_flat[split_0] - states_flat[split_1]) ** 2, dim=(1, 2))
                scaling_factor = 10
                loss_metric = torch.abs(dist_g * scaling_factor - dist_s).mean()

                g = g.view(torch.Size([bs, T + 1]) + g.size()[1:])

                """
                rollout 0 -> 1 : T + 1
                """
                U_for_pred = actions[:, : T]
                G_for_pred = model.simulate(T=T, g=g[:, 0], u_seq=U_for_pred, rel_attrs=rel_attrs[:, 0])

                ''' rollout time: T // 2 + 1, T '''
                data_for_ae = [x[:, :T + 1] for x in data]
                data_for_pred = [x[:, 1:T + 1] for x in data]

                # decode state for auto-encoding

                ''' BT x N x 4 '''
                attrs_for_ae_flat = get_flat(data_for_ae[0])
                rel_attrs_for_ae_flat = get_flat(data_for_ae[3])
                decode_s_for_ae = model.to_s(attrs=attrs_for_ae_flat, gcodes=get_flat(g[:, :T + 1]),
                                             rel_attrs=rel_attrs_for_ae_flat, pstep=args.pstep)

                # decode state for prediction

                ''' BT x N x 4 '''
                attrs_for_pred_flat = get_flat(data_for_pred[0])
                rel_attrs_for_pred_flat = get_flat(data_for_pred[3])
                decode_s_for_pred = model.to_s(attrs=attrs_for_pred_flat, gcodes=get_flat(G_for_pred),
                                               rel_attrs=rel_attrs_for_pred_flat, pstep=args.pstep)

                loss_auto_encode = F.l1_loss(
                    decode_s_for_ae, states[:, :T + 1].reshape(decode_s_for_ae.shape))
                loss_prediction = F.l1_loss(
                    decode_s_for_pred, states[:, 1:].reshape(decode_s_for_pred.shape))

                loss = loss_auto_encode + loss_prediction + loss_metric * args.lambda_loss_metric

                meter_loss_metric.update(loss_metric.item(), bs)
                meter_loss_ae.update(loss_auto_encode.item(), bs)
                meter_loss_pred.update(loss_prediction.item(), bs)

                meter_dist_g.update(dist_g.mean().item(), bs)
                meter_dist_s.update(dist_s.mean().item(), bs)

            '''prediction loss'''
            meter_loss.update(loss.item(), bs)

            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            if i % args.log_per_iter == 0:
                log = '%s [%d/%d][%d/%d] Loss: %.6f (%.6f), sysid_error: %.6f (%.6f), loss_ae: %.6f (%.6f), loss_pred: %.6f (%.6f), ' \
                      'loss_metric: %.6f (%.6f)' % (
                    phase, epoch, args.n_epoch, i, data_n_batches[phase],
                    loss.item(), meter_loss.avg,
                    fit_err.item(), meter_fit_error.avg,
                    loss_auto_encode.item(), meter_loss_ae.avg,
                    loss_prediction.item(), meter_loss_pred.avg,
                    loss_metric.item(), meter_loss_metric.avg,
                      )

                print()
                print(log)
                log_fout.write(log + '\n')
                log_fout.flush()

            if phase == 'train' and i % args.ckp_per_iter == 0:
                torch.save(model.state_dict(), '%s/net_epoch_%d_iter_%d.pth' % (args.outf, epoch, i))

        log = '%s [%d/%d] Loss: %.4f, Best valid: %.4f' % (phase, epoch, args.n_epoch, meter_loss.avg, best_valid_loss)
        print(log)
        log_fout.write(log + '\n')
        log_fout.flush()

        if phase == 'valid' and not args.eval:
            scheduler.step(meter_loss.avg)
            if meter_loss.avg < best_valid_loss:
                best_valid_loss = meter_loss.avg
                torch.save(model.state_dict(), '%s/net_best.pth' % (args.outf))

log_fout.close()
