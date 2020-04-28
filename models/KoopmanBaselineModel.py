import os

import numpy as np
import torch
from torch.autograd import Variable

from data import denormalize, normalize
from utils import load_data


class KoopmanBaseline(object):
    def __init__(self, args):
        self.args = args
        self.stat_path = os.path.join(self.args.dataf, 'stat.h5' if not args.demo else 'stat_demo.h5')
        self.stat = load_data(['attrs', 'states', 'actions'], self.stat_path)
        self.A = None
        self.B = None
        if args.fit_type == 'structured':
            self.system_identify = self.fit
            self.simulate = self.rollout
            self.step = self.linear_forward

    def to_s(self, attrs, gcodes, rel_attrs, pstep=None):
        """
        :param gcodes: B x N x G
        :return states: B x N x D
        """
        states = gcodes[:, :, :self.args.state_dim]
        if self.args.env in ['Soft', 'Swim']:
            states = self.regularize_state_Soft(states, rel_attrs, self.stat)
        return states

    def to_g(self, attrs, states, rel_attrs, pstep=None):
        """
        :param states: B x N x D
        :return gcodes: B x N x G
        """
        B, N, D = states.size()
        if self.args.env in ['Soft', 'Swim']:
            base = (states[:, :, :4] + states[:, :, 4:8] + states[:, :, 8:12] + states[:, :, 12:16]) / 4
        else:
            base = states

        one = states[:, :, :1].clone()
        one[:] = 1
        g_list = [states, one]

        for order in range(2, self.args.baseline_order + 1):
            # for j in range(1, order):
            #     poly = ((base[:, :, :, None] ** j) * (base[:, :, None, :] ** (order - j))).reshape(B, N, -1)
            #     g_list.append(poly)
            poly = ((base[:, :, :, None] ** (order - 1)) * (base[:, :, None, :] ** 1)).reshape(B, N, -1) / (3 ** order)
            g_list.append(poly)

        # base_square = (base[:,:,:,None] * base[:,:,None,:]).reshape(B,N,-1)
        # base_cube = ((base[:,:,:,None] ** 2) * base[:,:,None,:]).reshape(B,N,-1)

        # bavg = base.mean(1)[:,None,:].repeat(1,N,1)
        # base_avg = base * bavg

        # gcodes = torch.cat([states, base_square, base_cube], 2)
        gcodes = torch.cat(g_list, 2)
        return gcodes

    @staticmethod
    def get_aug(G, rel_attrs):
        """
        :param G: B x T x N x D
        :param rel_attrs:  B x N x N x R
        :return augG: B x T x N x R D
        """
        B, T, N, D = G.size()
        R = rel_attrs.size(-1)

        sumG_list = []
        for i in range(R):
            ''' B x T x N x N '''
            adj = rel_attrs[:, :, :, i][:, None, :, :].repeat(1, T, 1, 1)
            sumG = torch.bmm(
                adj.reshape(B * T, N, N),
                G.reshape(B * T, N, D)
            ).reshape(B, T, N, D)
            sumG_list.append(sumG)

        augG = torch.cat(sumG_list, 3)

        return augG

    def fit(self, G, H, U, rel_attrs, I_factor):
        """
        :param G: B x T x N x D
        :param H: B x T x N x D
        :param U: B x T x N x a_dim
        :param rel_attrs: B x N x N x R (relation_dim) rel_attrs[i,j] ==> receiver i, sender j
        :param I_factor: scalor
        :return:
        A: B x R D x D
        B: B x R a_dim x D
        s.t.
        H = augG @ A + augU @ B
        """

        ''' B x R: sqrt(# of appearance of block matrices of the same type)'''
        rel_weights = torch.sqrt(rel_attrs.sum(1).sum(1))
        rel_weights = torch.clamp(rel_weights, min=1e-8)

        bs, T, N, D = G.size()
        R = rel_attrs.size(-1)
        a_dim = U.size(3)

        ''' B x T x N x R D '''
        augG = self.get_aug(G, rel_attrs)
        ''' B x T x N x R a_dim'''
        augU = self.get_aug(U, rel_attrs)

        augG_reweight = augG.reshape(bs, T, N, R, D) / rel_weights[:, None, None, :, None]
        augU_reweight = augU.reshape(bs, T, N, R, a_dim) / rel_weights[:, None, None, :, None]

        ''' B x TN x R(D + a_dim)'''
        GU_reweight = torch.cat([augG_reweight.reshape(bs, T * N, R * D),
                                 augU_reweight.reshape(bs, T * N, R * a_dim)], 2)

        '''B x (R * D + R * a_dim) x D'''
        AB_reweight = torch.bmm(
            self.batch_pinv(GU_reweight, I_factor),
            H.reshape(bs, T * N, D)
        )
        self.A = AB_reweight[:, :R * D].reshape(bs, R, D, D) / rel_weights[:, :, None, None]
        self.B = AB_reweight[:, R * D:].reshape(bs, R, a_dim, D) / rel_weights[:, :, None, None]

        self.A = self.A.reshape(bs, R * D, D)
        self.B = self.B.reshape(bs, R * a_dim, D)

        fit_err = H.reshape(bs, T * N, D) - torch.bmm(GU_reweight, AB_reweight)
        fit_err = torch.sqrt((fit_err ** 2).mean())

        return self.A, self.B, fit_err

    def linear_forward(self, g, u, rel_attrs):
        """
        :param g: B x N x D
        :param u: B x N x a_dim
        :param rel_attrs: B x N x N x R
        :return B x N x D
        """
        ''' B x N x R D '''
        aug_g = self.get_aug(G=g[:, None, :, :], rel_attrs=rel_attrs)[:, 0]
        ''' B x N x R a_dim'''
        aug_u = self.get_aug(G=u[:, None, :, :], rel_attrs=rel_attrs)[:, 0]

        new_g = torch.bmm(aug_g, self.A) + torch.bmm(aug_u, self.B)
        return new_g

    def rollout(self, g, u_seq, T, rel_attrs):
        """
        :param g: B x N x D
        :param u_seq: B x T x N x a_dim
        :param rel_attrs: B x N x N x R
        :param T:
        :return:
        """
        g_list = []
        for t in range(T):
            g = self.linear_forward(g, u_seq[:, t], rel_attrs)
            g_list.append(g[:, None, :, :])
        return torch.cat(g_list, 1)

    @staticmethod
    def batch_pinv(x, I_factor):

        """
        :param x: B x N x D (N > D)
        :param I_factor:
        :return:
        """

        B, N, D = x.size()

        if N < D:
            x = torch.transpose(x, 1, 2)
            N, D = D, N
            trans = True
        else:
            trans = False

        x_t = torch.transpose(x, 1, 2)

        I = torch.eye(D)[None, :, :].repeat(B, 1, 1)
        use_gpu = torch.cuda.is_available()
        if use_gpu: I = I.cuda()

        x_pinv = torch.bmm(
            torch.inverse(torch.bmm(x_t, x) + I_factor * I),
            x_t
        )

        if trans:
            x_pinv = torch.transpose(x_pinv, 1, 2)

        return x_pinv

    @staticmethod
    def regularize_state_Soft(states, rel_attrs, stat):
        """
        :param states: B x N x state_dim
        :param rel_attrs: B x N x N x relation_dim
        :param stat: [xxx]
        :return new states: B x N x state_dim
        """
        states_denorm = denormalize([states], [stat[1]], var=True)[0]
        states_denorm_acc = denormalize([states.clone()], [stat[1]], var=True)[0]

        rel_attrs = rel_attrs[0]

        rel_attrs_np = rel_attrs.detach().cpu().numpy()

        def get_rel_id(x):
            return np.where(x > 0)[0][0]

        B, N, state_dim = states.size()
        count = Variable(torch.FloatTensor(np.zeros((1, N, 1, 8))).to(states.device))

        for i in range(N):
            for j in range(N):

                if i == j:
                    assert get_rel_id(rel_attrs_np[i, j]) % 9 == 0  # rel_attrs[i, j, 0] == 1
                    count[:, i, :, :] += 1
                    continue

                assert torch.sum(rel_attrs[i, j]) <= 1

                if torch.sum(rel_attrs[i, j]) == 0:
                    continue

                if get_rel_id(rel_attrs_np[i, j]) % 9 == 1:  # rel_attrs[i, j, 1] == 1:
                    assert get_rel_id(rel_attrs_np[j, i]) % 9 == 2  # rel_attrs[j, i, 2] == 1
                    x0 = 1;
                    y0 = 3
                    x1 = 0;
                    y1 = 2
                    idx = 1
                elif get_rel_id(rel_attrs_np[i, j]) % 9 == 2:  # rel_attrs[i, j, 2] == 1:
                    assert get_rel_id(rel_attrs_np[j, i]) % 9 == 1  # rel_attrs[j, i, 1] == 1
                    x0 = 3;
                    y0 = 1
                    x1 = 2;
                    y1 = 0
                    idx = 2
                elif get_rel_id(rel_attrs_np[i, j]) % 9 == 3:  # rel_attrs[i, j, 3] == 1:
                    assert get_rel_id(rel_attrs_np[j, i]) % 9 == 4  # rel_attrs[j, i, 4] == 1
                    x0 = 0;
                    y0 = 1
                    x1 = 2;
                    y1 = 3
                    idx = 3
                elif get_rel_id(rel_attrs_np[i, j]) % 9 == 4:  # rel_attrs[i, j, 4] == 1:
                    assert get_rel_id(rel_attrs_np[j, i]) % 9 == 3  # rel_attrs[j, i, 3] == 1
                    x0 = 1;
                    y0 = 0
                    x1 = 3;
                    y1 = 2
                    idx = 4
                elif get_rel_id(rel_attrs_np[i, j]) % 9 == 5:  # rel_attrs[i, j, 5] == 1:
                    assert get_rel_id(rel_attrs_np[j, i]) % 9 == 8  # rel_attrs[j, i, 8] == 1
                    x = 0;
                    y = 3
                    idx = 5
                elif get_rel_id(rel_attrs_np[i, j]) % 9 == 8:  # rel_attrs[i, j, 8] == 1:
                    assert get_rel_id(rel_attrs_np[j, i]) % 9 == 5  # rel_attrs[j, i, 5] == 1
                    x = 3;
                    y = 0
                    idx = 8
                elif get_rel_id(rel_attrs_np[i, j]) % 9 == 6:  # rel_attrs[i, j, 6] == 1:
                    assert get_rel_id(rel_attrs_np[j, i]) % 9 == 7  # rel_attrs[j, i, 7] == 1
                    x = 1;
                    y = 2
                    idx = 6
                elif get_rel_id(rel_attrs_np[i, j]) % 9 == 7:  # rel_attrs[i, j, 7] == 1:
                    assert get_rel_id(rel_attrs_np[j, i]) % 9 == 6  # rel_attrs[j, i, 6] == 1
                    x = 2;
                    y = 1
                    idx = 7
                else:
                    AssertionError("Unknown rel_attr %f" % rel_attrs[i, j])

                if idx < 5:
                    # if connect by two points
                    x0 *= 2;
                    y0 *= 2
                    x1 *= 2;
                    y1 *= 2
                    count[:, i, :, x0:x0 + 2] += 1
                    count[:, i, :, x1:x1 + 2] += 1
                    states_denorm_acc[:, i, x0:x0 + 2] += states_denorm[:, j, y0:y0 + 2]
                    states_denorm_acc[:, i, x0 + 8:x0 + 10] += states_denorm[:, j, y0 + 8:y0 + 10]
                    states_denorm_acc[:, i, x1:x1 + 2] += states_denorm[:, j, y1:y1 + 2]
                    states_denorm_acc[:, i, x1 + 8:x1 + 10] += states_denorm[:, j, y1 + 8:y1 + 10]

                else:
                    # if connected by a corner
                    x *= 2;
                    y *= 2
                    count[:, i, :, x:x + 2] += 1
                    states_denorm_acc[:, i, x:x + 2] += states_denorm[:, j, y:y + 2]
                    states_denorm_acc[:, i, x + 8:x + 10] += states_denorm[:, j, y + 8:y + 10]

        states_denorm = states_denorm_acc.view(B, N, 2, state_dim // 2) / count
        states_denorm = states_denorm.view(B, N, state_dim)

        return normalize([states_denorm], [stat[1]], var=True)[0]
