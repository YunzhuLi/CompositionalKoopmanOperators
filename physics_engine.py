import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pymunk
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Polygon
from pymunk.vec2d import Vec2d

from utils import rand_float, rand_int, calc_dis, norm


class Engine(object):
    def __init__(self, dt, state_dim, action_dim, param_dim):
        self.dt = dt
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.param_dim = param_dim

        self.state = None
        self.action = None
        self.param = None

        self.init()

    def init(self):
        pass

    def get_param(self):
        return self.param.copy()

    def set_param(self, param):
        self.param = param.copy()

    def get_state(self):
        return self.state.copy()

    def set_state(self, state):
        self.state = state.copy()

    def get_scene(self):
        return self.state.copy(), self.param.copy()

    def set_scene(self, state, param):
        self.state = state.copy()
        self.param = param.copy()

    def get_action(self):
        return self.action.copy()

    def set_action(self, action):
        self.action = action.copy()

    def d(self, state, t, param):
        # time derivative
        pass

    def step(self):
        pass

    def render(self, state, param):
        pass

    def clean(self):
        pass


class RopeEngine(Engine):

    def __init__(self, dt, state_dim, action_dim, param_dim,
                 num_mass_range=[4, 8], k_range=[500., 1500.], gravity_range=[-2., -8.],
                 position_range=[-0.6, 0.6], bihop=True):

        # state_dim = 4
        # action_dim = 1
        # param_dim = 5
        # param [n_ball, init_x, k, damping, gravity]

        self.radius = 0.06
        self.mass = 1.

        self.num_mass_range = num_mass_range
        self.k_range = k_range
        self.gravity_range = gravity_range
        self.position_range = position_range

        self.bihop = bihop

        super(RopeEngine, self).__init__(dt, state_dim, action_dim, param_dim)

    def init(self, param=None):
        if param is None:
            self.n_ball, self.init_x, self.k, self.damping, self.gravity = [None] * 5
        else:
            self.n_ball, self.init_x, self.k, self.damping, self.gravity = param
            self.n_ball = int(self.n_ball)

        num_mass_range = self.num_mass_range
        position_range = self.position_range
        if self.n_ball is None:
            self.n_ball = rand_int(num_mass_range[0], num_mass_range[1])
        if self.init_x is None:
            self.init_x = np.random.rand() * (position_range[1] - position_range[0]) + position_range[0]
        if self.k is None:
            self.k = rand_float(self.k_range[0], self.k_range[1])
        if self.damping is None:
            self.damping = self.k / 20.
        if self.gravity is None:
            self.gravity = rand_float(self.gravity_range[0], self.gravity_range[1])
        self.param = np.array([self.n_ball, self.init_x, self.k, self.damping, self.gravity])

        # print('Env Rope param: n_ball=%d, init_x=%.4f, k=%.4f, damping=%.4f, gravity=%.4f' % (
        #     self.n_ball, self.init_x, self.k, self.damping, self.gravity))

        self.space = pymunk.Space()
        self.space.gravity = (0., self.gravity)

        self.height = 1.0
        self.rest_len = 0.3

        self.add_masses()
        self.add_rels()

        self.state_prv = None

    @property
    def num_obj(self):
        return self.n_ball

    def add_masses(self):
        inertia = pymunk.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        x = self.init_x
        y = self.height
        self.balls = []

        for i in range(self.n_ball):
            body = pymunk.Body(self.mass, inertia)
            body.position = Vec2d(x, y)
            shape = pymunk.Circle(body, self.radius, (0, 0))

            if i == 0:
                # fix the first mass to a specific height
                move_joint = pymunk.GrooveJoint(self.space.static_body, body, (-2, y), (2, y), (0, 0))
                self.space.add(body, shape, move_joint)
            else:
                self.space.add(body, shape)

            self.balls.append(body)
            y -= self.rest_len

    def add_rels(self):
        give = 1. + 0.075
        # add springs over adjacent balls
        for i in range(self.n_ball - 1):
            c = pymunk.DampedSpring(
                self.balls[i], self.balls[i + 1], (0, 0), (0, 0),
                rest_length=self.rest_len * give, stiffness=self.k, damping=self.damping)
            self.space.add(c)

        # add bihop springs
        if self.bihop:
            for i in range(self.n_ball - 2):
                c = pymunk.DampedSpring(
                    self.balls[i], self.balls[i + 2], (0, 0), (0, 0),
                    rest_length=self.rest_len * give * 2, stiffness=self.k * 0.5, damping=self.damping)
                self.space.add(c)

    def add_impulse(self):
        impulse = (self.action[0], 0)
        self.balls[0].apply_impulse_at_local_point(impulse=impulse, point=(0, 0))

    def get_param(self):
        return self.n_ball, self.init_x, self.k, self.damping, self.gravity

    def get_state(self):
        state = np.zeros((self.n_ball, 4))
        for i in range(self.n_ball):
            ball = self.balls[i]
            state[i] = np.array([ball.position[0], ball.position[1], ball.velocity[0], ball.velocity[1]])

        vel_dim = self.state_dim // 2
        if self.state_prv is None:
            state[:, vel_dim:] = 0
        else:
            state[:, vel_dim:] = (state[:, :vel_dim] - self.state_prv[:, :vel_dim]) / self.dt

        return state

    def step(self):
        self.add_impulse()
        self.state_prv = self.get_state()
        self.space.step(self.dt)

    def render(self, states, actions=None, param=None, video=True, image=False, path=None,
               act_scale=None, draw_edge=True, lim=(-2.5, 2.5, -2.5, 2.5), states_gt=None,
               count_down=False, gt_border=False):
        if video:
            video_path = path + '.avi'
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            print('Save video as %s' % video_path)
            out = cv2.VideoWriter(video_path, fourcc, 25, (640, 480))

        if image:
            image_path = path + '_img'
            print('Save images to %s' % image_path)
            os.system('mkdir -p %s' % image_path)

        c = ['royalblue', 'tomato', 'limegreen', 'orange', 'violet', 'chocolate', 'lightsteelblue']

        time_step = states.shape[0]
        n_ball = states.shape[1]

        if actions is not None and actions.ndim == 3:
            '''get the first ball'''
            actions = actions[:, 0, :]

        for i in range(time_step):
            fig, ax = plt.subplots(1)
            plt.xlim(lim[0], lim[1])
            plt.ylim(lim[2], lim[3])
            plt.axis('off')

            if draw_edge:
                cnt = 0
                for x in range(n_ball - 1):
                    plt.plot([states[i, x, 0], states[i, x + 1, 0]],
                             [states[i, x, 1], states[i, x + 1, 1]],
                             '-', color=c[1], lw=2, alpha=0.5)

            circles = []
            circles_color = []
            for j in range(n_ball):
                circle = Circle((states[i, j, 0], states[i, j, 1]), radius=self.radius * 5 / 4)
                circles.append(circle)
                circles_color.append(c[0])

            pc = PatchCollection(circles, facecolor=circles_color, linewidth=0, alpha=1.)
            ax.add_collection(pc)

            if states_gt is not None:
                circles = []
                circles_color = []
                for j in range(n_ball):
                    circle = Circle((states_gt[i, j, 0], states_gt[i, j, 1]), radius=self.radius * 5 / 4)
                    circles.append(circle)
                    circles_color.append('orangered')
                pc = PatchCollection(circles, facecolor=circles_color, linewidth=0, alpha=1.)
                ax.add_collection(pc)

            if actions is not None:
                F = actions[i, 0] / 4
                normF = norm(F)
                if normF < 1e-10:
                    pass
                else:
                    ax.arrow(states[i, 0, 0] + F / normF * 0.1, states[i, 0, 1],
                             F, 0., fc='Orange', ec='Orange', width=0.04, head_width=0.2, head_length=0.2)

            ax.set_aspect('equal')

            font = {'family': 'serif',
                    'color': 'darkred',
                    'weight': 'normal',
                    'size': 16}
            if count_down:
                plt.text(-2.5, 1.5, 'CountDown: %d' % (time_step - i - 1), fontdict=font)

            plt.tight_layout()

            if video:
                fig.canvas.draw()
                frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
                if i == time_step - 1:
                    for _ in range(5):
                        out.write(frame)

            if image:
                plt.savefig(os.path.join(image_path, 'fig_%s.png' % i), bbox_inches='tight')

            plt.close()

        if video:
            out.release()


# ===================================================================
'''
For Soft and Swim
'''


def get_init_p_fish_8():
    init_p = np.zeros((8, 3))
    init_p[0, :] = np.array([0, 0, 2])
    init_p[1, :] = np.array([0, 1, 0])
    init_p[2, :] = np.array([0, 2, 2])
    init_p[3, :] = np.array([0, 3, 0])
    init_p[4, :] = np.array([1, 0, 2])
    init_p[5, :] = np.array([1, 1, 0])
    init_p[6, :] = np.array([1, 2, 2])
    init_p[7, :] = np.array([1, 3, 0])
    return init_p


def sample_init_p_flight(n_box, shape_type=None, aug=False, train=False,
                         min_offset=False, max_offset=False):
    assert 5 <= n_box < 10
    c_box_dict = {
        5: [[1, 3, 1], [2, 1, 2]],
        6: [[3, 3], [2, 2, 2]],
        7: [[2, 3, 2], [1, 2, 1, 2, 1], [2, 1, 1, 1, 2]],
        8: [[2, 2, 2, 2], [1, 2, 2, 2, 1], [2, 1, 2, 1, 2], [3, 2, 3]],
        9: [[2, 2, 1, 2, 2], [1, 2, 3, 2, 1], [2, 1, 3, 1, 2], [3, 3, 3]],
    }

    if shape_type is None:
        shape_type = rand_int(0, len(c_box_dict[n_box]))
    else:
        shape_type = shape_type % len(c_box_dict[n_box])

    c_box = c_box_dict[n_box][shape_type]

    init_p = np.zeros((n_box, 3))
    y_offset = np.zeros(len(c_box))

    for i in range(1, (len(c_box) + 1) // 2):
        left = c_box[i - 1]
        right = c_box[i]
        y_offset[i] = rand_int(1 - right, left)
        if min_offset: y_offset[i] = 1 - right
        if max_offset: y_offset[i] = left
        y_offset[len(c_box) - i] = - y_offset[i]
        assert len(c_box) - i > i

    y = np.zeros(len(c_box))
    for i in range(1, len(c_box)):
        y[i] = y[i - 1] + y_offset[i]
    y -= y.min()

    # print('y_offset', y_offset, 'y', y)

    while True:
        idx = 0
        for i, c in enumerate(c_box):
            for j in range(c):
                # if not train:
                if False:
                    material = 2 if j < c - 1 or c == 1 else 0
                else:
                    r = np.random.rand()
                    if c == 1:
                        r_actuated, r_soft, r_rigid = 0.25, 0.25, 0.5
                    elif j == 0:
                        r_actuated, r_soft, r_rigid = 0.0, 0.5, 0.5
                    elif j == c - 1:
                        r_actuated, r_soft, r_rigid = 0.75, 0.25, 0.0
                    else:
                        r_actuated, r_soft, r_rigid = 0.4, 0.2, 0.4
                    if r < r_actuated:
                        material = 0
                    elif r < r_actuated + r_soft:
                        material = 1
                    else:
                        material = 2
                init_p[idx, :] = np.array([i, y[i] + j, material])
                idx += 1

        if (init_p[:, 2] == 0).sum() >= 2:
            break

    # print('init_p', init_p)

    if aug:
        if np.random.rand() > 0.5:
            '''flip y'''
            init_p[:, 1] = -init_p[:, 1]
        if np.random.rand() > 0.5:
            '''flip x'''
            init_p[:, 0] = -init_p[:, 0]
        if np.random.rand() > 0.5:
            '''swap x and y'''
            x, y = init_p[:, 0], init_p[:, 1]
            init_p[:, 0], init_p[:, 1] = y.copy(), x.copy()

    # print('init_p', init_p)

    return init_p


def sample_init_p_regular(n_box, shape_type=None, aug=False):
    print('sample_init_p')
    init_p = np.zeros((n_box, 3))

    if shape_type is None: shape_type = rand_int(0, 4)
    print('shape_type', shape_type)

    if shape_type == 0:  # 0 or u shape
        init_p[0, :] = np.array([0, 0, 2])
        init_p[1, :] = np.array([-1, 0, 2])
        init_p[2, :] = np.array([1, 0, 2])
        idx = 3
        y = 0
        x = [-1, 0, 1]
        res = n_box - 3
        while res > 0:
            y += 1
            if res == 3:
                i_list = [0, 1, 2]
            else:
                i_list = [0, 2]
            material = [0, 1][int(np.random.rand() < 0.5 and res > 3)]
            for i in i_list:
                init_p[idx, :] = np.array([x[i], y, material])
                idx += 1
                res -= 1

    elif shape_type == 1:  # 1 shape
        init_p[0, :] = np.array([0, 0, 2])
        for i in range(1, n_box):
            material = [0, 1][int(np.random.rand() < 0.5 and i < n_box - 1)]
            init_p[i, :] = np.array([0, i, material])

    elif shape_type == 2:  # I shape
        if n_box < 7:
            init_p[0, :] = np.array([0, 0, 2])
            for i in range(1, n_box - 3):
                material = [0, 1][int(np.random.rand() < 0.5 and i < n_box - 1)]
                init_p[i, :] = np.array([0, i, material])
            init_p[n_box - 1, :] = np.array([-1, n_box - 3, 0])
            init_p[n_box - 2, :] = np.array([0, n_box - 3, 0])
            init_p[n_box - 3, :] = np.array([1, n_box - 3, 0])
        else:
            init_p[0, :] = np.array([-1, 0, 2])
            init_p[1, :] = np.array([0, 0, 2])
            init_p[2, :] = np.array([1, 0, 2])
            for i in range(3, n_box - 3):
                material = [0, 1][int(np.random.rand() < 0.5 and i < n_box - 1)]
                init_p[i, :] = np.array([0, i - 2, material])
            init_p[n_box - 1, :] = np.array([-1, n_box - 5, 0])
            init_p[n_box - 2, :] = np.array([0, n_box - 5, 0])
            init_p[n_box - 3, :] = np.array([1, n_box - 5, 0])

    elif shape_type == 3:  # T shape
        if n_box < 6:
            init_p[0, :] = np.array([-1, 0, 2])
            init_p[1, :] = np.array([0, 0, 2])
            init_p[2, :] = np.array([1, 0, 2])
            for i in range(3, n_box):
                material = [0, 1][int(np.random.rand() < 0.5 and i < n_box - 1)]
                init_p[i, :] = np.array([0, i - 2, material])
        else:
            init_p[0, :] = np.array([-2, 0, 2])
            init_p[1, :] = np.array([-1, 0, 2])
            init_p[2, :] = np.array([0, 0, 2])
            init_p[3, :] = np.array([1, 0, 2])
            init_p[4, :] = np.array([2, 0, 2])
            for i in range(5, n_box):
                material = [0, 1][int(np.random.rand() < 0.5 and i < n_box - 1)]
                init_p[i, :] = np.array([0, i - 4, material])

    elif shape_type == 4:  # stronger T
        assert n_box == 10
        init_p[0, :] = np.array([0, -4, 0])
        init_p[1, :] = np.array([1, -4, 1])
        init_p[2, :] = np.array([0, -3, 0])
        init_p[3, :] = np.array([1, -3, 0])
        init_p[4, :] = np.array([0, -2, 1])
        init_p[5, :] = np.array([1, -2, 0])
        init_p[6, :] = np.array([-1, -1, 2])
        init_p[7, :] = np.array([0, -1, 2])
        init_p[8, :] = np.array([1, -1, 2])
        init_p[9, :] = np.array([2, -1, 2])

    if aug:
        if np.random.rand() > 0.5:
            '''flip y'''
            init_p[:, 1] = -init_p[:, 1]
        if np.random.rand() > 0.5:
            '''swap x and y'''
            x, y = init_p[:, 0], init_p[:, 1]
            init_p[:, 0], init_p[:, 1] = y.copy(), x.copy()

    return init_p


class SoftEngine(Engine):

    def __init__(self, dt, state_dim, action_dim, param_dim,
                 num_box_range=[5, 10], k_range=[600, 1000.]):

        # state_dim = 4
        # action_dim = 1
        # param_dim = 4 - [n_box, k, damping, init_p]
        # init_p: n_box * 3 - [x, y, type]
        # type: 0 - soft & actuated, 1 - soft, 2 - rigid

        self.side_length = 1.
        self.num_box_range = num_box_range
        self.k_range = k_range
        self.radius = 0.01
        self.mass = 1.

        super(SoftEngine, self).__init__(dt, state_dim, action_dim, param_dim)

    @property
    def num_obj(self):
        return self.n_box

    def inside_lim(self, x, y, lim):
        if x >= lim[0] and x < lim[1] and y >= lim[0] and y < lim[1]:
            return True
        return False

    def sample_init_p(self):
        n_box = self.n_box
        r_actuated = 0.5
        r_soft = 0.25
        r_rigid = 0.25
        lim = -4, 4
        mask = np.zeros((lim[1] - lim[0], lim[1] - lim[0]))

        init_p = np.zeros((n_box, 3))
        buf = []

        # add a fixed box
        x, y = 0, -4
        init_p[0] = np.array([x, y, 3])
        buf.append([x - 1, y])
        buf.append([x, y + 1])
        buf.append([x + 1, y])
        mask[x, y] = mask[x - 1, y] = mask[x, y + 1] = mask[x + 1, y] = 1

        for i in range(1, n_box):
            roll_type = np.random.rand()
            if roll_type < r_actuated:
                init_p[i, 2] = 0
            elif roll_type < r_actuated + r_soft:
                init_p[i, 2] = 1
            else:
                init_p[i, 2] = 2

            if len(buf) > 0:
                idx = rand_int(0, len(buf))
                x = buf[idx][0]
                y = buf[idx][1]
                del buf[idx]
            else:
                x = rand_int(lim[0], lim[1])
                y = rand_int(lim[0], lim[1])

            init_p[i, 0], init_p[i, 1] = x, y

            mask[x, y] = 1
            if self.inside_lim(x + 1, y, lim) and mask[x + 1, y] == 0:
                buf.append([x + 1, y]);
                mask[x + 1, y] = 1
            if self.inside_lim(x - 1, y, lim) and mask[x - 1, y] == 0:
                buf.append([x - 1, y]);
                mask[x - 1, y] = 1
            if self.inside_lim(x, y + 1, lim) and mask[x, y + 1] == 0:
                buf.append([x, y + 1]);
                mask[x, y + 1] = 1
            if self.inside_lim(x, y - 1, lim) and mask[x, y - 1] == 0:
                buf.append([x, y - 1]);
                mask[x, y - 1] = 1

        while (init_p[:, 2] == 0).sum() < 2:
            ''' less than 2 actuated'''
            ''' re-generate box type'''
            for i in range(1, n_box):
                roll_type = np.random.rand()
                if roll_type < r_actuated:
                    init_p[i, 2] = 0
                elif roll_type < r_actuated + r_soft:
                    init_p[i, 2] = 1
                else:
                    init_p[i, 2] = 2

        return init_p

    def init(self, param=None):
        if param is None:
            self.n_box, self.k, self.damping, self.init_p = [None] * 4
        else:
            self.n_box, self.k, self.damping, self.init_p = param
            self.n_box = int(self.n_box)

        if self.n_box is None:
            self.n_box = rand_int(self.num_box_range[0], self.num_box_range[1])
        if self.k is None:
            self.k = rand_float(self.k_range[0], self.k_range[1])
        if self.damping is None:
            self.damping = self.k / 20.
        if self.init_p is None:
            self.init_p = self.sample_init_p()
            # self.init_p = sample_init_p_regular(self.n_box, shape_type=4)

        # print('Env Soft param: n_box=%d, k=%.4f, damping=%.4f' % (self.n_box, self.k, self.damping))

        self.space = pymunk.Space()
        self.space.gravity = (0., 0.)

        self.add_masses()
        self.add_rels()

        self.state_prv = None

    def add_masses(self):
        inertia = pymunk.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        self.balls = []

        for i in range(self.n_box):
            x, y, t = self.init_p[i]
            l = self.side_length / 2.

            for j in range(4):
                body = pymunk.Body(self.mass, inertia)

                if j == 0:
                    body.position = Vec2d(x - l, y - l)
                elif j == 1:
                    body.position = Vec2d(x - l, y + l)
                elif j == 2:
                    body.position = Vec2d(x + l, y - l)
                else:
                    body.position = Vec2d(x + l, y + l)

                # shape = pymunk.Circle(body, self.radius, (0, 0))
                # self.space.add(body, shape)
                self.space.add(body)
                self.balls.append(body)

    def add_rels(self):
        ball = self.balls[0]
        c = pymunk.PinJoint(self.space.static_body, ball, (ball.position[0], ball.position[1]), (0, 0))
        self.space.add(c)
        ball = self.balls[2]
        c = pymunk.PinJoint(self.space.static_body, ball, (ball.position[0], ball.position[1]), (0, 0))
        self.space.add(c)
        c = pymunk.DampedSpring(
            self.balls[0], self.balls[1], (0, 0), (0, 0),
            rest_length=self.side_length, stiffness=self.k, damping=self.damping)
        self.space.add(c)
        c = pymunk.DampedSpring(
            self.balls[1], self.balls[3], (0, 0), (0, 0),
            rest_length=self.side_length, stiffness=self.k, damping=self.damping)
        self.space.add(c)
        c = pymunk.DampedSpring(
            self.balls[2], self.balls[3], (0, 0), (0, 0),
            rest_length=self.side_length, stiffness=self.k, damping=self.damping)
        self.space.add(c)
        c = pymunk.DampedSpring(
            self.balls[1], self.balls[2], (0, 0), (0, 0),
            rest_length=self.side_length * np.sqrt(2), stiffness=self.k, damping=self.damping)
        self.space.add(c)
        c = pymunk.DampedSpring(
            self.balls[0], self.balls[3], (0, 0), (0, 0),
            rest_length=self.side_length * np.sqrt(2), stiffness=self.k, damping=self.damping)
        self.space.add(c)

        for i in range(1, self.n_box):
            if self.init_p[i, 2] <= 1:
                # if the box is soft
                # side
                c = pymunk.DampedSpring(
                    self.balls[i * 4], self.balls[i * 4 + 1], (0, 0), (0, 0),
                    rest_length=self.side_length, stiffness=self.k, damping=self.damping)
                self.space.add(c)
                c = pymunk.DampedSpring(
                    self.balls[i * 4], self.balls[i * 4 + 2], (0, 0), (0, 0),
                    rest_length=self.side_length, stiffness=self.k, damping=self.damping)
                self.space.add(c)
                c = pymunk.DampedSpring(
                    self.balls[i * 4 + 3], self.balls[i * 4 + 1], (0, 0), (0, 0),
                    rest_length=self.side_length, stiffness=self.k, damping=self.damping)
                self.space.add(c)
                c = pymunk.DampedSpring(
                    self.balls[i * 4 + 3], self.balls[i * 4 + 2], (0, 0), (0, 0),
                    rest_length=self.side_length, stiffness=self.k, damping=self.damping)
                self.space.add(c)
                # cross
                c = pymunk.DampedSpring(
                    self.balls[i * 4], self.balls[i * 4 + 3], (0, 0), (0, 0),
                    rest_length=self.side_length * np.sqrt(2), stiffness=self.k, damping=self.damping)
                self.space.add(c)
                c = pymunk.DampedSpring(
                    self.balls[i * 4 + 1], self.balls[i * 4 + 2], (0, 0), (0, 0),
                    rest_length=self.side_length * np.sqrt(2), stiffness=self.k, damping=self.damping)
                self.space.add(c)
            else:
                # if the box is rigid
                # side
                c = pymunk.PinJoint(self.balls[i * 4], self.balls[i * 4 + 1], (0, 0), (0, 0))
                self.space.add(c)
                c = pymunk.PinJoint(self.balls[i * 4], self.balls[i * 4 + 2], (0, 0), (0, 0))
                self.space.add(c)
                c = pymunk.PinJoint(self.balls[i * 4 + 3], self.balls[i * 4 + 1], (0, 0), (0, 0))
                self.space.add(c)
                c = pymunk.PinJoint(self.balls[i * 4 + 3], self.balls[i * 4 + 2], (0, 0), (0, 0))
                self.space.add(c)
                # cross
                c = pymunk.PinJoint(self.balls[i * 4], self.balls[i * 4 + 3], (0, 0), (0, 0))
                self.space.add(c)
                c = pymunk.PinJoint(self.balls[i * 4 + 1], self.balls[i * 4 + 2], (0, 0), (0, 0))
                self.space.add(c)

        # add PinJoint to adjacent boxes
        for i in range(self.n_box):
            for j in range(i):
                for ii in range(4):
                    for jj in range(4):
                        x, y = i * 4 + ii, j * 4 + jj
                        if calc_dis(self.balls[x].position, self.balls[y].position) < 1e-4:
                            c = pymunk.PinJoint(self.balls[x], self.balls[y], (0, 0), (0, 0))
                            self.space.add(c)

    def add_force(self):
        for i in range(self.n_box):
            if self.init_p[i, 2] == 0:
                # if the current box has actuator
                for j in range(4):
                    x, y = i * 4 + j, i * 4 + (3 - j)
                    direct = np.array([
                        self.balls[y].position[0] - self.balls[x].position[0],
                        self.balls[y].position[1] - self.balls[x].position[1]])
                    direct /= norm(direct)
                    force = direct * self.action[i]
                    self.balls[x].apply_force_at_local_point(
                        force=(force[0], force[1]), point=(0, 0))

    def get_param(self):
        return self.n_box, self.k, self.damping, self.init_p

    def get_state(self):
        state = np.zeros((self.n_box, 16))
        for i in range(self.n_box):
            for j in range(4):
                ball = self.balls[i * 4 + j]
                state[i, j * 2: (j + 1) * 2] = \
                    np.array([ball.position[0], ball.position[1]])
                state[i, 8 + j * 2: 8 + (j + 1) * 2] = \
                    np.array([ball.velocity[0], ball.velocity[1]])

        state_acc = state.copy()
        count = np.zeros((self.n_box, 1, 8))

        for i in range(self.n_box):
            for j in range(self.n_box):
                if i == j:
                    count[i, :, :] += 1
                    continue

                delta = self.init_p[i, :2] - self.init_p[j, :2]

                assert (np.abs(delta) > 0).any()

                if (np.abs(delta) > 1).any():
                    # no contact
                    continue

                if np.sum(np.abs(delta)) == 1:
                    # contact at a side
                    if delta[0] == 1:
                        x0, y0, x1, y1 = 1, 3, 0, 2
                    elif delta[0] == -1:
                        x0, y0, x1, y1 = 3, 1, 2, 0
                    elif delta[1] == 1:
                        x0, y0, x1, y1 = 0, 1, 2, 3
                    elif delta[1] == -1:
                        x0, y0, x1, y1 = 1, 0, 3, 2

                    x0 *= 2
                    y0 *= 2
                    x1 *= 2
                    y1 *= 2
                    count[i, :, x0:x0 + 2] += 1
                    count[i, :, x1:x1 + 2] += 1
                    state_acc[i, x0:x0 + 2] += state[j, y0:y0 + 2]
                    state_acc[i, x0 + 8:x0 + 10] += state[j, y0 + 8:y0 + 10]
                    state_acc[i, x1:x1 + 2] += state[j, y1:y1 + 2]
                    state_acc[i, x1 + 8:x1 + 10] += state[j, y1 + 8:y1 + 10]

                elif np.sum(np.abs(delta)) == 2:
                    # contact at a corner
                    if delta[0] == 1 and delta[1] == 1:
                        x, y = 0, 3
                    elif delta[0] == 1 and delta[1] == -1:
                        x, y = 1, 2
                    elif delta[0] == -1 and delta[1] == 1:
                        x, y = 2, 1
                    elif delta[0] == -1 and delta[1] == -1:
                        x, y = 3, 0

                    x *= 2
                    y *= 2
                    count[i, :, x:x + 2] += 1
                    state_acc[i, x:x + 2] += state[j, y:y + 2]
                    state_acc[i, x + 8:x + 10] += state[j, y + 8:y + 10]

        state_acc = state_acc.reshape(self.n_box, 2, 8) / count
        state_acc = state_acc.reshape(self.n_box, 16)

        vel_dim = self.state_dim // 2
        if self.state_prv is None:
            state_acc[:, vel_dim:] = 0
        else:
            state_acc[:, vel_dim:] = (state_acc[:, :vel_dim] - self.state_prv[:, :vel_dim]) / self.dt

        return state_acc

    def step(self):
        self.add_force()
        self.state_prv = self.get_state()
        self.space.step(self.dt)

    def render(self, states, actions=None, param=None, act_scale=10.,
               video=True, image=False, path=None, lim=(-5., 5., -6., 4.),
               states_gt=None, count_down=False, gt_border=False):

        if video:
            video_path = path + '.avi'
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            print('Save video as %s' % video_path)
            out = cv2.VideoWriter(video_path, fourcc, 25, (640, 480))

        if image:
            image_path = path + '_img'
            print('Save images to %s' % image_path)
            os.system('mkdir -p %s' % image_path)

        c = ['royalblue', 'tomato', 'limegreen', 'orange', 'violet', 'chocolate', 'lightsteelblue']

        time_step = states.shape[0]
        n_ball = states.shape[1] * 4
        states = states[:, :, :8].reshape((time_step, n_ball, 2))

        if states_gt is not None:
            states_gt = states_gt[:, :, :8].reshape((time_step, n_ball, 2))

        init_p = param[3]

        for i in range(time_step):
            fig, ax = plt.subplots(1)
            plt.xlim(lim[0], lim[1])
            plt.ylim(lim[2], lim[3])
            plt.axis('off')

            polys = []
            polys_color = []

            circles = []
            circles_color = []

            for j in [0, 2]:
                circle = Circle((states[i, j, 0], states[i, j, 1]), radius=0.1)
                circles.append(circle)
                circles_color.append('orangered')

            for j in range(self.n_box):
                poly = Polygon(np.array([
                    states[i, j * 4, :2], states[i, j * 4 + 1, :2],
                    states[i, j * 4 + 3, :2], states[i, j * 4 + 2, :2]]), True)
                polys.append(poly)

                if init_p[j, 2] == 0:
                    if actions is not None:
                        act = actions[i, j]
                    else:
                        act = 0.
                    r = (act + act_scale) / (act_scale * 2)
                    if np.abs(r - 0.5) < 1e-4:
                        c = 'cornflowerblue'
                    else:
                        c = to_rgba('tomato')[:3] * r + to_rgba('limegreen')[:3] * (1. - r)
                        c = np.clip(c, 0., 1.)
                    polys_color.append(c)

                elif init_p[j, 2] == 1:
                    polys_color.append('lightsteelblue')
                elif init_p[j, 2] == 2:
                    polys_color.append('dimgray')
                elif init_p[j, 2] == 3:
                    polys_color.append('lightsteelblue')
                else:
                    raise AssertionError("Unknown box type %f" % init_p[j, 2])

            if states_gt is not None:
                polys_gt = []
                for j in range(self.n_box):
                    poly = Polygon(np.array([
                        states_gt[i, j * 4, :2], states_gt[i, j * 4 + 1, :2],
                        states_gt[i, j * 4 + 3, :2], states_gt[i, j * 4 + 2, :2]]), True)
                    polys_gt.append(poly)

                if gt_border:
                    pc_polys_gt = PatchCollection(
                        polys_gt, facecolor=(0., 0., 0., 0.), edgecolor='orangered', lw=1.)
                else:
                    pc_polys_gt = PatchCollection(
                        polys_gt, facecolor=polys_color, linewidth=0, alpha=0.5)

                circles_gt = []
                for j in [0, 2]:
                    circle = Circle((states[i, j, 0], states[i, j, 1]), radius=0.1)
                    circles_gt.append(circle)

                pc_circles_gt = PatchCollection(circles_gt, facecolor=circles_color, linewidth=0, alpha=0.5)

            pc_polys = PatchCollection(polys, facecolor=polys_color, linewidth=0, alpha=1.)
            pc_circles = PatchCollection(circles, facecolor=circles_color, linewidth=0, alpha=1.)

            ax.add_collection(pc_polys)
            ax.add_collection(pc_circles)

            if states_gt is not None:
                ax.add_collection(pc_polys_gt)
                ax.add_collection(pc_circles_gt)

            ax.set_aspect('equal')

            font = {'family': 'serif',
                    'color': 'darkred',
                    'weight': 'normal',
                    'size': 16}
            if count_down:
                plt.text(-5, 3, 'CountDown: %d' % (time_step - i - 1), fontdict=font)

            plt.tight_layout()

            if video:
                fig.canvas.draw()
                frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
                if i == time_step - 1:
                    for _ in range(10):
                        out.write(frame)

            if image:
                plt.savefig(os.path.join(image_path, 'fig_%s.png' % i), bbox_inches='tight')

            plt.close()

        if video:
            out.release()


class SwimEngine(Engine):

    def __init__(self, dt, state_dim, action_dim, param_dim,
                 num_box_range=[5, 10], k_range=[600, 800.]):

        # state_dim = 4
        # action_dim = 1
        # param_dim = 4 - [n_box, k, damping, init_p]
        # init_p: n_box * 3 - [x, y, type]
        # type: 0 - soft & actuated, 1 - soft, 2 - rigid

        self.side_length = 1.
        self.num_box_range = num_box_range
        self.k_range = k_range
        self.radius = 0.01
        self.mass = 1.

        super(SwimEngine, self).__init__(dt, state_dim, action_dim, param_dim)

    @property
    def num_obj(self):
        return self.n_box

    def inside_lim(self, x, y, lim):
        if x >= lim[0] and x < lim[1] and y >= lim[0] and y < lim[1]:
            return True
        return False

    def sample_init_p(self):
        n_box = self.n_box
        r_actuated = 0.5
        r_soft = 0.25
        r_rigid = 0.25
        lim = -4, 4
        mask = np.zeros((lim[1] - lim[0], lim[1] - lim[0]))

        init_p = np.zeros((n_box, 3))
        buf = []

        for i in range(n_box):
            roll_type = np.random.rand()
            if roll_type < r_actuated:
                init_p[i, 2] = 0
            elif roll_type < r_actuated + r_soft:
                init_p[i, 2] = 1
            else:
                init_p[i, 2] = 2

            if len(buf) > 0:
                idx = rand_int(0, len(buf))
                x = buf[idx][0]
                y = buf[idx][1]
                del buf[idx]
            else:
                x = rand_int(lim[0] // 2, lim[1] // 2)
                y = rand_int(lim[0] // 2, lim[1] // 2)

            init_p[i, 0], init_p[i, 1] = x, y

            mask[x, y] = 1
            if self.inside_lim(x + 1, y, lim) and mask[x + 1, y] == 0:
                buf.append([x + 1, y]);
                mask[x + 1, y] = 1
            if self.inside_lim(x - 1, y, lim) and mask[x - 1, y] == 0:
                buf.append([x - 1, y]);
                mask[x - 1, y] = 1
            if self.inside_lim(x, y + 1, lim) and mask[x, y + 1] == 0:
                buf.append([x, y + 1]);
                mask[x, y + 1] = 1
            if self.inside_lim(x, y - 1, lim) and mask[x, y - 1] == 0:
                buf.append([x, y - 1]);
                mask[x, y - 1] = 1

        while (init_p[:, 2] == 0).sum() < 2:
            ''' less than 2 actuated'''
            ''' re-generate box type'''
            for i in range(n_box):
                roll_type = np.random.rand()
                if roll_type < r_actuated:
                    init_p[i, 2] = 0
                elif roll_type < r_actuated + r_soft:
                    init_p[i, 2] = 1
                else:
                    init_p[i, 2] = 2

        return init_p

    def calc_outside(self):
        # recorde whether a specific edge is in the outside
        self.outside = np.ones((self.n_box, 4))
        for i in range(self.n_box):
            for j in range(self.n_box):
                if i == j:
                    continue

                delta = self.init_p[i, :2] - self.init_p[j, :2]

                assert (np.abs(delta) > 0).any()

                if (np.abs(delta) > 1).any():
                    # no contact
                    continue

                if np.sum(np.abs(delta)) == 1:
                    # contact at a side
                    if delta[0] == 1:
                        self.outside[i, 0] = 0
                    elif delta[0] == -1:
                        self.outside[i, 2] = 0
                    elif delta[1] == 1:
                        self.outside[i, 3] = 0
                    elif delta[1] == -1:
                        self.outside[i, 1] = 0

    def init(self, param=None):
        if param is None:
            self.n_box, self.k, self.damping, self.init_p = [None] * 4
        else:
            self.n_box, self.k, self.damping, self.init_p = param
            self.n_box = int(self.n_box)

        if self.n_box is None:
            self.n_box = rand_int(self.num_box_range[0], self.num_box_range[1])
        if self.k is None:
            self.k = rand_float(self.k_range[0], self.k_range[1])
        if self.damping is None:
            self.damping = self.k / 20.
        if self.init_p is None:
            self.init_p = self.sample_init_p()

        # print('Env Swim param: n_box=%d, k=%.4f, damping=%.4f' % (self.n_box, self.k, self.damping))

        self.space = pymunk.Space()
        self.space.gravity = (0., 0.)

        self.add_masses()
        self.add_rels()
        self.calc_outside()

        self.state_prv = None

        # print(self.init_p)
        # print(self.outside)

    def add_masses(self):
        inertia = pymunk.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        self.balls = []

        for i in range(self.n_box):
            x, y, t = self.init_p[i]
            l = self.side_length / 2.

            for j in range(4):
                body = pymunk.Body(self.mass, inertia)

                if j == 0:
                    body.position = Vec2d(x - l, y - l)
                elif j == 1:
                    body.position = Vec2d(x - l, y + l)
                elif j == 2:
                    body.position = Vec2d(x + l, y - l)
                else:
                    body.position = Vec2d(x + l, y + l)

                # shape = pymunk.Circle(body, self.radius, (0, 0))
                # self.space.add(body, shape)
                self.space.add(body)
                self.balls.append(body)

    def add_rels(self):
        for i in range(self.n_box):
            if self.init_p[i, 2] <= 1:
                # if the box is soft
                # side
                c = pymunk.DampedSpring(
                    self.balls[i * 4], self.balls[i * 4 + 1], (0, 0), (0, 0),
                    rest_length=self.side_length, stiffness=self.k, damping=self.damping)
                self.space.add(c)
                c = pymunk.DampedSpring(
                    self.balls[i * 4], self.balls[i * 4 + 2], (0, 0), (0, 0),
                    rest_length=self.side_length, stiffness=self.k, damping=self.damping)
                self.space.add(c)
                c = pymunk.DampedSpring(
                    self.balls[i * 4 + 3], self.balls[i * 4 + 1], (0, 0), (0, 0),
                    rest_length=self.side_length, stiffness=self.k, damping=self.damping)
                self.space.add(c)
                c = pymunk.DampedSpring(
                    self.balls[i * 4 + 3], self.balls[i * 4 + 2], (0, 0), (0, 0),
                    rest_length=self.side_length, stiffness=self.k, damping=self.damping)
                self.space.add(c)
                # cross
                c = pymunk.DampedSpring(
                    self.balls[i * 4], self.balls[i * 4 + 3], (0, 0), (0, 0),
                    rest_length=self.side_length * np.sqrt(2), stiffness=self.k, damping=self.damping)
                self.space.add(c)
                c = pymunk.DampedSpring(
                    self.balls[i * 4 + 1], self.balls[i * 4 + 2], (0, 0), (0, 0),
                    rest_length=self.side_length * np.sqrt(2), stiffness=self.k, damping=self.damping)
                self.space.add(c)
            else:
                # if the box is rigid
                # side
                c = pymunk.PinJoint(self.balls[i * 4], self.balls[i * 4 + 1], (0, 0), (0, 0))
                self.space.add(c)
                c = pymunk.PinJoint(self.balls[i * 4], self.balls[i * 4 + 2], (0, 0), (0, 0))
                self.space.add(c)
                c = pymunk.PinJoint(self.balls[i * 4 + 3], self.balls[i * 4 + 1], (0, 0), (0, 0))
                self.space.add(c)
                c = pymunk.PinJoint(self.balls[i * 4 + 3], self.balls[i * 4 + 2], (0, 0), (0, 0))
                self.space.add(c)
                # cross
                c = pymunk.PinJoint(self.balls[i * 4], self.balls[i * 4 + 3], (0, 0), (0, 0))
                self.space.add(c)
                c = pymunk.PinJoint(self.balls[i * 4 + 1], self.balls[i * 4 + 2], (0, 0), (0, 0))
                self.space.add(c)

        # add PinJoint to adjacent boxes
        for i in range(self.n_box):
            for j in range(i):
                for ii in range(4):
                    for jj in range(4):
                        x, y = i * 4 + ii, j * 4 + jj
                        if calc_dis(self.balls[x].position, self.balls[y].position) < 1e-4:
                            c = pymunk.PinJoint(self.balls[x], self.balls[y], (0, 0), (0, 0))
                            self.space.add(c)

    def add_force(self):
        for i in range(self.n_box):
            if self.init_p[i, 2] == 0:
                # if the current box has actuator
                for j in range(4):
                    x, y = i * 4 + j, i * 4 + (3 - j)
                    direct = np.array([
                        self.balls[y].position[0] - self.balls[x].position[0],
                        self.balls[y].position[1] - self.balls[x].position[1]])
                    direct /= norm(direct)
                    force = direct * self.action[i]
                    self.balls[x].apply_force_at_local_point(
                        force=(force[0], force[1]), point=(0, 0))

        for i in range(self.n_box):
            s = np.zeros((4, 4))
            for j in range(4):
                idx = i * 4 + j
                s[j, 0] = self.balls[idx].position[0]
                s[j, 1] = self.balls[idx].position[1]
                s[j, 2] = self.balls[idx].velocity[0]
                s[j, 3] = self.balls[idx].velocity[1]

            for j in range(4):
                if j == 0:
                    a, b = 0, 1
                elif j == 1:
                    a, b = 1, 3
                elif j == 2:
                    a, b = 3, 2
                else:
                    a, b = 2, 0

                if self.outside[i, j] == 1 and self.init_p[i, 2] == 0 and self.action[i] < 0:
                    direct = s[b, :2] - s[a, :2]
                    dist = norm(direct)
                    direct /= dist
                    direct = np.array([-direct[1], direct[0]])

                    v_scale = np.dot(s[a, 2:], direct)
                    if v_scale > 0.:
                        f = - v_scale ** 2 * direct * dist * 50.
                        self.balls[i * 4 + a].apply_force_at_local_point(
                            force=(f[0], f[1]), point=(0, 0))

                    v_scale = np.dot(s[b, 2:], direct)
                    if v_scale > 0.:
                        f = - v_scale ** 2 * direct * dist * 50.
                        self.balls[i * 4 + b].apply_force_at_local_point(
                            force=(f[0], f[1]), point=(0, 0))

    def get_param(self):
        return self.n_box, self.k, self.damping, self.init_p

    def get_state(self):
        state = np.zeros((self.n_box, 16))
        for i in range(self.n_box):
            for j in range(4):
                ball = self.balls[i * 4 + j]
                state[i, j * 2: (j + 1) * 2] = \
                    np.array([ball.position[0], ball.position[1]])
                state[i, 8 + j * 2: 8 + (j + 1) * 2] = \
                    np.array([ball.velocity[0], ball.velocity[1]])

        state_acc = state.copy()
        count = np.zeros((self.n_box, 1, 8))

        for i in range(self.n_box):
            for j in range(self.n_box):
                if i == j:
                    count[i, :, :] += 1
                    continue

                delta = self.init_p[i, :2] - self.init_p[j, :2]

                assert (np.abs(delta) > 0).any()

                if (np.abs(delta) > 1).any():
                    # no contact
                    continue

                if np.sum(np.abs(delta)) == 1:
                    # contact at a side
                    if delta[0] == 1:
                        x0, y0, x1, y1 = 1, 3, 0, 2
                    elif delta[0] == -1:
                        x0, y0, x1, y1 = 3, 1, 2, 0
                    elif delta[1] == 1:
                        x0, y0, x1, y1 = 0, 1, 2, 3
                    elif delta[1] == -1:
                        x0, y0, x1, y1 = 1, 0, 3, 2

                    x0 *= 2
                    y0 *= 2
                    x1 *= 2
                    y1 *= 2
                    count[i, :, x0:x0 + 2] += 1
                    count[i, :, x1:x1 + 2] += 1
                    state_acc[i, x0:x0 + 2] += state[j, y0:y0 + 2]
                    state_acc[i, x0 + 8:x0 + 10] += state[j, y0 + 8:y0 + 10]
                    state_acc[i, x1:x1 + 2] += state[j, y1:y1 + 2]
                    state_acc[i, x1 + 8:x1 + 10] += state[j, y1 + 8:y1 + 10]

                elif np.sum(np.abs(delta)) == 2:
                    # contact at a corner
                    if delta[0] == 1 and delta[1] == 1:
                        x, y = 0, 3
                    elif delta[0] == 1 and delta[1] == -1:
                        x, y = 1, 2
                    elif delta[0] == -1 and delta[1] == 1:
                        x, y = 2, 1
                    elif delta[0] == -1 and delta[1] == -1:
                        x, y = 3, 0

                    x *= 2
                    y *= 2
                    count[i, :, x:x + 2] += 1
                    state_acc[i, x:x + 2] += state[j, y:y + 2]
                    state_acc[i, x + 8:x + 10] += state[j, y + 8:y + 10]

        state_acc = state_acc.reshape(self.n_box, 2, 8) / count
        state_acc = state_acc.reshape(self.n_box, 16)

        vel_dim = self.state_dim // 2
        if self.state_prv is None:
            state_acc[:, vel_dim:] = 0
        else:
            state_acc[:, vel_dim:] = (state_acc[:, :vel_dim] - self.state_prv[:, :vel_dim]) / self.dt

        return state_acc

    def step(self):
        self.add_force()
        self.state_prv = self.get_state()
        self.space.step(self.dt)

    def render(self, states, actions=None, param=None, act_scale=10.,
               video=True, image=False, path=None, lim=(-6., 6., -7., 5.),
               states_gt=None, count_down=False, gt_border=False):

        if video:
            video_path = path + '.avi'
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            print('Save video as %s' % video_path)
            out = cv2.VideoWriter(video_path, fourcc, 25, (640, 480))

        if image:
            image_path = path + '_img'
            print('Save images to %s' % image_path)
            os.system('mkdir -p %s' % image_path)

        c = ['royalblue', 'tomato', 'limegreen', 'orange', 'violet', 'chocolate', 'lightsteelblue']

        time_step = states.shape[0]
        n_ball = states.shape[1] * 4
        states = states[:, :, :8].reshape((time_step, n_ball, 2))

        if states_gt is not None:
            states_gt = states_gt[:, :, :8].reshape((time_step, n_ball, 2))

        init_p = param[3]

        for i in range(time_step):
            fig, ax = plt.subplots(1)
            plt.xlim(lim[0], lim[1])
            plt.ylim(lim[2], lim[3])
            plt.axis('off')

            polys = []
            polys_color = []

            for j in range(self.n_box):
                poly = Polygon(np.array([
                    states[i, j * 4, :2], states[i, j * 4 + 1, :2],
                    states[i, j * 4 + 3, :2], states[i, j * 4 + 2, :2]]), True)
                polys.append(poly)

                if init_p[j, 2] == 0:
                    if actions is not None:
                        act = actions[i, j]
                    else:
                        act = 0.
                    r = (act + act_scale) / (act_scale * 2)
                    if np.abs(r - 0.5) < 1e-4:
                        c = 'cornflowerblue'
                    else:
                        c = to_rgba('tomato')[:3] * r + to_rgba('limegreen')[:3] * (1. - r)
                        c = np.clip(c, 0., 1.)
                    polys_color.append(c)

                elif init_p[j, 2] == 1:
                    polys_color.append('lightsteelblue')
                elif init_p[j, 2] == 2:
                    polys_color.append('dimgray')
                elif init_p[j, 2] == 3:
                    polys_color.append('lightsteelblue')
                else:
                    raise AssertionError("Unknown box type %f" % init_p[j, 2])

            if states_gt is not None:
                polys_gt = []
                for j in range(self.n_box):
                    poly = Polygon(np.array([
                        states_gt[i, j * 4, :2], states_gt[i, j * 4 + 1, :2],
                        states_gt[i, j * 4 + 3, :2], states_gt[i, j * 4 + 2, :2]]), True)
                    polys_gt.append(poly)

                if gt_border:
                    pc_polys_gt = PatchCollection(
                        polys_gt, facecolor=(0., 0., 0., 0.), edgecolor='orangered', lw=1.)
                else:
                    pc_polys_gt = PatchCollection(
                        polys_gt, facecolor=polys_color, linewidth=0, alpha=0.5)

            pc_polys = PatchCollection(polys, facecolor=polys_color, linewidth=0, alpha=1.)

            ax.add_collection(pc_polys)

            if states_gt is not None:
                ax.add_collection(pc_polys_gt)

            ax.set_aspect('equal')

            font = {'family': 'serif',
                    'color': 'darkred',
                    'weight': 'normal',
                    'size': 16}
            if count_down:
                plt.text(-7, 4, 'CountDown: %d' % (time_step - i - 1), fontdict=font)

            plt.tight_layout()

            if video:
                fig.canvas.draw()
                frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
                if i == time_step - 1:
                    for _ in range(10):
                        out.write(frame)

            if image:
                plt.savefig(os.path.join(image_path, 'fig_%s.png' % i), bbox_inches='tight')

            plt.close()

        if video:
            out.release()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='')
    args = parser.parse_args()

    os.system('mkdir -p test')

    if args.env == 'Rope':
        dt = 1. / 50.
        state_dim = 4
        action_dim = 1
        param_dim = 5  # n_ball, init_x, k, damping, gravity

        act_scale = 2.
        ret_scale = 1.

        engine = RopeEngine(dt, state_dim, action_dim, param_dim)

        time_step = 300
        states = np.zeros((time_step, engine.n_ball, engine.state_dim))
        actions = np.zeros((time_step, engine.action_dim))

        for i in range(time_step):
            states[i] = engine.get_state()
            act = (np.random.rand() * 2. - 1.) * act_scale - states[i, 0, 0] * ret_scale
            engine.set_action(np.array([act]))
            engine.step()
            actions[i] = engine.get_action()

        engine.render(states, None, engine.get_param(), video=True, image=True, path='test/Rope')

    elif args.env == 'Soft':
        dt = 1. / 50.
        state_dim = 16
        action_dim = 1
        param_dim = 4  # n_box, k, damping, init_p

        act_scale = 800.
        act_delta = 200.

        engine = SoftEngine(dt, state_dim, action_dim, param_dim)
        engine.init()

        time_step = 100
        states = np.zeros((time_step, engine.n_box, state_dim))
        actions = np.zeros((time_step, engine.n_box, action_dim))

        for i in range(time_step):
            states[i] = engine.get_state()
            box_type = engine.init_p[:, 2]
            for j in range(engine.n_box):
                if box_type[j] == 0:
                    # if this is a actuated box
                    if i == 0:
                        actions[i, j] = rand_float(-act_delta, act_delta)
                    else:
                        actions[i, j] = actions[i - 1, j] + rand_float(-act_delta, act_delta)
                        actions[i, j] = np.clip(actions[i, j], -act_scale, act_scale)
                elif box_type[j] >= 1:
                    # if this is a soft box without actuation OR a rigid box
                    actions[i, j] = 0

            engine.set_action(actions[i])
            engine.step()
            assert np.array_equal(actions[i], engine.get_action())

        engine.render(states, None, engine.get_param(), act_scale=act_scale, video=True, image=True, path='test/Soft',
                      count_down=False)

    elif args.env == 'Swim':
        dt = 1. / 50.
        state_dim = 16
        action_dim = 1
        param_dim = 4  # n_box, k, damping, init_p

        act_scale = 600.
        act_delta = 300.

        engine = SwimEngine(dt, state_dim, action_dim, param_dim)

        tag = ['rand', 'forward', 'rotate'][0]
        for epoch in range(5):
            for num in [8]:
                init_p = sample_init_p_flight(num, epoch, True, train=False)
                engine.init(param=[num, None, None, init_p])

                '''
                init_p = get_init_p_fish_8()
                engine.init(param=[8, None, None, init_p])
                '''

                time_step = 100
                states = np.zeros((time_step, engine.n_box, state_dim))
                actions = np.zeros((time_step, engine.n_box, action_dim))
                actions_param = np.zeros((engine.n_box, 3))

                sin_motion = np.random.rand() < 0.5

                for i in range(time_step):
                    states[i] = engine.get_state()
                    box_type = engine.init_p[:, 2]
                    for j in range(engine.n_box):
                        if box_type[j] == 0:
                            # if this is a actuated box
                            if i == 0:
                                actions_param[j] = np.array(
                                    [rand_float(0., 1.), rand_float(0.5, 4.), rand_float(0, np.pi * 2)])

                            if actions_param[j, 0] < 0.5 and sin_motion == 0:
                                if i == 0:
                                    actions[i, j] = rand_float(-act_delta, act_delta)
                                else:
                                    lo = max(actions[i - 1, j] - act_delta, -act_scale)
                                    hi = min(actions[i - 1, j] + act_delta, act_scale)
                                    actions[i, j] = rand_float(lo, hi)
                                    actions[i, j] = np.clip(actions[i, j], -act_scale, act_scale)
                            else:
                                actions[i, j] = np.sin(i / actions_param[j, 1] + actions_param[j, 2]) * \
                                                rand_float(act_scale / 2., act_scale)

                            if tag == 'rotate':
                                if j < engine.n_box // 2:
                                    if actions[i, j] < 0: actions[i, j] = 0
                                else:
                                    if actions[i, j] > 0: actions[i, j] = 0

                        elif box_type[j] >= 1:
                            # if this is a soft box without actuation OR a rigid box
                            actions[i, j] = 0

                    engine.set_action(actions[i])
                    engine.step()
                    assert np.array_equal(actions[i], engine.get_action())

                os.system('mkdir -p test/swim_{}_train'.format(tag))
                engine.render(
                    states, None, engine.get_param(), act_scale=act_scale, video=True, image=True,
                    path='test/swim_{}_train/Swim_{}_{}'.format(tag, num, epoch), count_down=False)
