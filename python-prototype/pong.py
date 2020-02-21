#
# Pong: The Machine Learning Game
#
# Requirements:
# * Python 3
# * pip install pygame pymunk pytorch
# Note: Use venv, as it is easy to mess things up doing it for the first time.
#

import os
import gc
import sys
import copy
import math
import getopt
import random
from collections import namedtuple

# Pygame
# Set of Python modules designed for writing video games
import pygame
from pygame import Vector2, Rect

# Pymunk
# 2D physics library, a pythonic wrapper for Chipmunk written in C
#
import pymunk
import pymunk.util
import pymunk.pygame_util
from pymunk import Vec2d

# PyTorch
# Machine Learning framework
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import tensorboard

import matplotlib.pyplot as plt

SIMPLE = False

BACKGROUND_COLOR    = (192, 192, 192)
PRIMARY_COLOR       = (96, 96, 96)
TIMESTEP = 1/30     # Timestep is fixed to make simulation more deterministic and independent of frame-per-seconds.
GRAVITY = 0.085
BOARD_WIDTH = 1
BOARD_HEIGHT = 2
BALL_RADIUS = 0.0227
BALL_MASS = 1
SLIDER_Y = 0.9
MOUNTING_RADIUS = 0.0135
MOUNTING_MASS = 6
PADDLE_LENGTH = 0.14
PADDLE_SHANK_RATIO = 0.35
PADDLE_THICKNESS = PADDLE_LENGTH / 10
PADDLE_MASS = 6
PADDLE_ANGULAR_VELOCITY_DAMPING = 0.95
PADDLE_FORCE_MULTIPLIER = 6

if SIMPLE:
    MOUNTING_RADIUS = 0.065

class Transform:
    def __init__(self, scale, offset):
        self.scale = scale
        self.offset = offset

    def fw(self, v):
        if isinstance(v, (Vector2, Vec2d)):
            return Vector2(v.x * self.scale + self.offset.x, v.y * self.scale + self.offset.y)
        elif isinstance(v, (int, float)):
            return v * self.scale
        elif isinstance(v, tuple) and len(v) == 4:
            ulp = self.fw(Vector2(v[0], v[1]))
            return Rect(ulp.x, ulp.y, self.fw(v[2]), self.fw(v[3]))
        else:
            assert(False)

    def fw_int(self, v):
        v = self.fw(v)
        if isinstance(v, (Vector2, Vec2d)):
            return (int(v.x), int(v.y))
        elif isinstance(v, (int, float)):
            return int(v)
        elif isinstance(v, Rect):
            return (int(v.left), int(v.top), int(v.width), int(v.height))
        else:
            assert(False)

class Ball:
    def __init__(self, board):
        inertia = pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS, (0, 0))
        self.body = pymunk.Body(BALL_MASS, inertia)
        self.shape = pymunk.Circle(self.body, BALL_RADIUS, Vec2d(0, 0))
        self.shape.elasticity = 1
        self.shape.friction = 1
        board.space.add(self.body, self.shape)

class Paddle:
    def __init__(self, board, slot):
        assert(slot in [0, 1])
        self.slot = slot
        self.slot_multiplier = [1, -1][self.slot]

        inertia = pymunk.moment_for_circle(MOUNTING_MASS, 0, MOUNTING_RADIUS, (0, 0))
        self.mounting_body = pymunk.Body(MOUNTING_MASS, inertia)
        self.mounting_body.position = Vec2d(0, self.slot_multiplier * SLIDER_Y)
        self.mounting_shape = pymunk.Circle(self.mounting_body, MOUNTING_RADIUS * 0.95, Vec2d(0, 0))
        self.mounting_shape.elasticity = 0.5
        self.mounting_shape.friction = 1
        board.space.add(self.mounting_body, self.mounting_shape)

        if SIMPLE:
            return

        points = [
            Vec2d(-PADDLE_THICKNESS / 2, -0.5 * PADDLE_LENGTH),
            Vec2d( PADDLE_THICKNESS / 2, -0.5 * PADDLE_LENGTH),
            Vec2d( PADDLE_THICKNESS / 2,  0.5 * PADDLE_LENGTH),
            Vec2d(-PADDLE_THICKNESS / 2,  0.5 * PADDLE_LENGTH) ]

        intertia = pymunk.moment_for_poly(PADDLE_MASS, points, (0, 0))
        self.paddle_body = pymunk.Body(PADDLE_MASS, intertia)
        self.paddle_body.position = Vec2d(0, self.slot_multiplier * (SLIDER_Y + MOUNTING_RADIUS + 0.5 * PADDLE_LENGTH))
        self.paddle_shape = pymunk.Poly(self.paddle_body, points)
        self.paddle_shape.friction = 1
        self.paddle_shape.elasticity = 1

        joint = pymunk.PinJoint(self.mounting_body, self.paddle_body, Vec2d(0, 0), Vec2d(0, self.slot_multiplier * (-MOUNTING_RADIUS - 0.5 * PADDLE_LENGTH)))

        board.space.add(self.paddle_body, self.paddle_shape, joint)

class Board:
    def __init__(self, starting_slot, first_hit_wins):
        self.time = 0
        self.winner = None
        self.curr_hit = None
        self.first_hit_wins = first_hit_wins

        self.space = pymunk.Space()
        self.space.gravity = 0, GRAVITY if starting_slot == 0 else -GRAVITY

        self.wall_body = pymunk.Body(body_type=pymunk.Body.STATIC)

        left_points = [
            Vec2d(-0.5, -1),
            Vec2d(-1,   -1),
            Vec2d(-1,    1),
            Vec2d(-0.5,  1) ]
        right_points = [
            Vec2d( 0.5, -1),
            Vec2d( 1,   -1),
            Vec2d( 1,    1),
            Vec2d( 0.5,  1) ]

        self.left_wall_shape = pymunk.Poly(self.wall_body, left_points)
        self.left_wall_shape.friction = 1
        self.left_wall_shape.elasticity = 0.65

        self.right_wall_shape = pymunk.Poly(self.wall_body, right_points)
        self.right_wall_shape.friction = self.left_wall_shape.friction
        self.right_wall_shape.elasticity = self.left_wall_shape.elasticity

        self.space.add(self.wall_body, self.left_wall_shape, self.right_wall_shape)

        self.ball = Ball(self)
        self.ball.body.position = Vec2d(0.75 * (random.random() - 0.5), 0)
        self.ball.body.velocity = Vec2d(0.2 * (random.random() - 0.5), 0)

        self.paddles = [Paddle(self, 0), Paddle(self, 1)]

        for slot in [0, 1]:
            paddle = self.paddles[slot]
            joint = pymunk.GrooveJoint(self.wall_body, paddle.mounting_body,
                groove_a = Vec2d(-0.5, paddle.slot_multiplier * SLIDER_Y),
                groove_b = Vec2d(0.5, paddle.slot_multiplier * SLIDER_Y),
                anchor_b = Vec2d(0, 0) )
            self.space.add(joint)

        self.handler = self.space.add_default_collision_handler()
        self.handler.post_solve = self._post_solve

    def _post_solve(self, arbiter, space, data):
        if SIMPLE:
            if self.ball.shape in arbiter.shapes:
                if self.paddles[0].mounting_shape in arbiter.shapes:
                    if self.space.gravity.y > 0:
                        self.curr_hit = 0
                        self.curr_rewards[0] += 50 + 37.5 * -self.ball.body.velocity.y
                    self.space.gravity = 0, -GRAVITY
                elif self.paddles[1].mounting_shape in arbiter.shapes:
                    if self.space.gravity.y < 0:
                        self.curr_hit = 1
                        self.curr_rewards[1] += 50 + 37.5 * self.ball.body.velocity.y
                    self.space.gravity = 0, GRAVITY
            return

        if self.ball.shape in arbiter.shapes:
            if self.paddles[0].mounting_shape in arbiter.shapes:
                self.winner = 1
                self.curr_rewards[0] -= 50
                self.curr_rewards[1] += 100
            elif self.paddles[1].mounting_shape in arbiter.shapes:
                self.winner = 0
                self.curr_rewards[0] -= 100
                self.curr_rewards[1] -= 50
            elif self.paddles[0].paddle_shape in arbiter.shapes:
                if self.space.gravity.y > 0:
                    self.curr_hit = 0
                    self.curr_rewards[0] += 50 + 37.5 * -self.ball.body.velocity.y
                self.space.gravity = 0, -GRAVITY
            elif self.paddles[1].paddle_shape in arbiter.shapes:
                if self.space.gravity.y < 0:
                    self.curr_hit = 1
                    self.curr_rewards[1] += 50 + 37.5 * self.ball.body.velocity.y
                self.space.gravity = 0, GRAVITY

    def draw(self, screen, transform):
        pygame.draw.rect(screen, PRIMARY_COLOR, transform.fw_int((-0.5, -1, 1, 2)), 1)
        pygame.draw.circle(screen, PRIMARY_COLOR, transform.fw_int(self.ball.body.position), transform.fw_int(BALL_RADIUS), 1)

        for slot in [0, 1]:
            paddle = self.paddles[slot]

            pos_m = transform.fw_int(paddle.mounting_body.local_to_world(Vec2d(0, 0)))
            pygame.draw.circle(screen, PRIMARY_COLOR, pos_m, transform.fw_int(MOUNTING_RADIUS), 1)

            if not SIMPLE:
                pos_0 = transform.fw_int(paddle.paddle_body.local_to_world(Vec2d(-0.5 * PADDLE_THICKNESS, -0.5 * PADDLE_LENGTH)))
                pos_1 = transform.fw_int(paddle.paddle_body.local_to_world(Vec2d(-0.5 * PADDLE_THICKNESS, +0.5 * PADDLE_LENGTH)))
                pos_2 = transform.fw_int(paddle.paddle_body.local_to_world(Vec2d(+0.5 * PADDLE_THICKNESS, -0.5 * PADDLE_LENGTH)))
                pos_3 = transform.fw_int(paddle.paddle_body.local_to_world(Vec2d(+0.5 * PADDLE_THICKNESS, +0.5 * PADDLE_LENGTH)))
                pygame.draw.aaline(screen, PRIMARY_COLOR, pos_0, pos_1)
                pygame.draw.aaline(screen, PRIMARY_COLOR, pos_2, pos_3)
                pygame.draw.aaline(screen, PRIMARY_COLOR, pos_0, pos_2)
                pygame.draw.aaline(screen, PRIMARY_COLOR, pos_1, pos_3)

    def update(self, actions):
        assert(len(actions) == 2 and actions[0] in [0, 1] and actions[1] in [0, 1])
        for slot in [0, 1]:
            action = actions[slot]
            paddle = self.paddles[slot]

            force = Vec2d((2 * action - 1) * PADDLE_FORCE_MULTIPLIER * paddle.slot_multiplier, 0)
            pos = paddle.mounting_body.local_to_world(Vec2d(0, 0.0))
            paddle.mounting_body.apply_force_at_world_point(force, pos)

            paddle.mounting_body.angular_velocity = 0
            if not SIMPLE:
                paddle.paddle_body.angular_velocity *= PADDLE_ANGULAR_VELOCITY_DAMPING

        self.curr_hit = None
        self.curr_rewards = [0, 0]

        self.space.step(TIMESTEP)
        self._check_winner()
        self.curr_rewards = [float(self.curr_rewards[0]), float(self.curr_rewards[1])]

        self.time += 1

        return self.curr_rewards

    def _check_winner(self):
        if self.winner is not None:
            return

        if self.time > 3500:
            self.winner = 0   # 0 is incorrect here
            self.curr_rewards[0] = 0
            self.curr_rewards[1] = 0
            return

        if self.first_hit_wins and self.curr_hit is not None:
            self.winner = self.curr_hit
            self.curr_rewards[self.winner] += 50
            self.curr_rewards[1 - self.winner] -= 100
        else:
            ball_y = self.ball.body.position.y
            if ball_y > 1:
                self.winner = 1
            elif ball_y < -1:
                self.winner = 0

            if self.winner is not None:
                self.curr_rewards[self.winner] += 100
                self.curr_rewards[1 - self.winner] -= 50

    def state(self, slot):
        terminal = self.winner is not None
        slot_multiplier = [1, -1][slot]
        gravity = slot_multiplier * (1 if self.space.gravity.y > 0 else -1)

        if SIMPLE:
            ball_pos = slot_multiplier * self.ball.body.position
            ball_vel = slot_multiplier * self.ball.body.velocity

            my_paddle = self.paddles[slot]
            my_paddle_pos = slot_multiplier * my_paddle.mounting_body.position
            my_paddle_vel = slot_multiplier * my_paddle.mounting_body.velocity

            x = torch.tensor([
                gravity,
                ball_pos.x, ball_pos.y,
                ball_vel.x, ball_vel.y,
                my_paddle_pos.x, my_paddle_pos.y,
                my_paddle_vel.x, my_paddle_vel.y,
            ], device=device, dtype=torch.float32)
            return x, terminal

        ball_pos = slot_multiplier * self.ball.body.position
        ball_vel = slot_multiplier * self.ball.body.velocity
        ball_angle = slot_multiplier * self.ball.body.angle
        ball_ang_vel = slot_multiplier * self.ball.body.angular_velocity

        my_paddle = self.paddles[slot]
        my_paddle_pos = slot_multiplier * my_paddle.mounting_body.position
        my_paddle_vel = slot_multiplier * my_paddle.mounting_body.velocity
        my_paddle_angle = slot_multiplier * my_paddle.paddle_body.angle
        my_paddle_angle_sin = math.sin(my_paddle_angle)
        my_paddle_angle_cos = math.cos(my_paddle_angle)
        my_paddle_ang_vel = slot_multiplier * my_paddle.paddle_body.angular_velocity

        # foe_paddle = self.paddles[1 - slot]
        # foe_paddle_pos = slot_multiplier * foe_paddle.mounting_body.position
        # foe_paddle_vel = slot_multiplier * foe_paddle.mounting_body.velocity
        # foe_paddle_angle = slot_multiplier * foe_paddle.paddle_body.angle
        # foe_paddle_angle_sin = math.sin(foe_paddle_angle)
        # foe_paddle_angle_cos = math.sin(foe_paddle_angle)
        # foe_paddle_ang_vel = slot_multiplier * foe_paddle.paddle_body.angular_velocity

        def psdiv(x):
            x = float(x)
            if x > 0:
                return 1/(0.5 + x)
            else:
                return 1/(-0.5 + x)

        x = torch.tensor([
            gravity,
            ball_pos.x, ball_pos.y,                         psdiv(ball_pos.x), psdiv(ball_pos.y),
            ball_vel.x, ball_vel.y,                         psdiv(1/ball_vel.x), psdiv(ball_vel.y),
            #ball_angle, ball_ang_vel,                       psdiv(ball_angle), psdiv(ball_ang_vel),
            #ball_angle, ball_ang_vel,                       psdiv(ball_angle), psdiv(ball_ang_vel),
            my_paddle_pos.x, my_paddle_pos.y,               psdiv(my_paddle_pos.x), psdiv(my_paddle_pos.y),
            my_paddle_vel.x, my_paddle_vel.y,               psdiv(my_paddle_vel.x), psdiv(my_paddle_vel.y),
            my_paddle_angle_sin, my_paddle_angle_cos,       psdiv(my_paddle_angle_sin), psdiv(my_paddle_angle_cos),
            my_paddle_ang_vel,                              psdiv(my_paddle_ang_vel)
            # foe_paddle_pos.x, foe_paddle_pos.y,
            # foe_paddle_vel.x, foe_paddle_vel.y,
            # foe_paddle_angle_sin, foe_paddle_angle_cos,
            # foe_paddle_ang_vel,
        ], device=device, dtype=torch.float32)
        return x, terminal

def get_in_feature_count():
    return Board(0, False).state(0)[0].size(0)

class HumanPlayer:
    def __init__(self, slot):
        self.name = "human"
        self.slot = slot
    
    def new_game(self, board, match_num):
        self.board = board

    def __call__(self):
        pressed_keys = pygame.key.get_pressed()

        horizontal_change = 0
        if pressed_keys[pygame.K_LEFT if self.slot == 0 else pygame.K_d]:
            horizontal_change -= 1
        if pressed_keys[pygame.K_RIGHT if self.slot == 0 else pygame.K_a]:
            horizontal_change += 1

        if horizontal_change == 0:
            horizontal_change = 2 * (self.board.time % 2) - 1

        return {-1: 0, 1: 1}[horizontal_change]

    def reward(self, reward):
        pass

class PassivePlayer:
    def __init__(self, slot):
        self.name = "environment"
    
    def new_game(self, board, match_num):
        self.board = board

    def __call__(self):
        return self.board.time % 2

    def reward(self, reward):
        pass

class Game:
    def __init__(self, players_desc, display, vsync):
        assert(len(players_desc) == 2)
        self.display = display
        self.vsync = vsync and self.display
        self.match_num = -1
        self.wins = [0, 0]

        learn = "h" not in players_desc
        self.players = [None, None]
        for slot in range(2):
            if players_desc[slot] == "h":
                player = HumanPlayer(slot)
                assert(self.display)
                self.vsync = True
            elif players_desc[slot] == "e":
                player = PassivePlayer(slot)
            else:
                player = Bot(slot, players_desc[slot], learn=learn)
            self.players[slot] = player

        if self.display:
            pygame.init()
            pygame.display.set_caption("Gravi-Pong")
            self.screen = pygame.display.set_mode(
                size=(340, 640),
                flags=pygame.DOUBLEBUF)

            self.transform = Transform(scale=300, offset=Vector2(170, 320))
            self.clock = pygame.time.Clock()

        self._new_game()

    def _new_game(self):
        self.match_num += 1

        if isinstance(self.players[0], (HumanPlayer, Bot)):
            if isinstance(self.players[1], (HumanPlayer, Bot)):
                starting_slot = self.match_num % 2
                first_hit_wins = False
            else:
                starting_slot = 0
                first_hit_wins = True
        else:
            if isinstance(self.players[1], (HumanPlayer, Bot)):
                starting_slot = 1
                first_hit_wins = True
            else:
                starting_slot = self.match_num % 2
                first_hit_wins = False

        self.board = Board(starting_slot, first_hit_wins)
        for player in self.players:
            player.new_game(self.board, self.match_num)

        print("Starting match {}".format(self.match_num))
        gc.collect()

    def run(self):
        self.running = True

        while self.running:
            if self.display:
                self._draw()
                self._process_events()
            self._update()

    def _draw(self):
        self.screen.fill(BACKGROUND_COLOR)
        self.board.draw(self.screen, self.transform)
        pygame.display.flip()
        if self.vsync:
            self.clock.tick(1 / TIMESTEP)

    def _process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def _update(self):
        player_actions = [self.players[slot]() for slot in range(2)]

        rewards = self.board.update(player_actions)
        if rewards[0] != 0 or rewards[1] != 0:
            print("Rewards: " + str(rewards))

        for slot in range(2):
            self.players[slot].reward(rewards[slot])

        winner = self.board.winner
        if winner is not None:
            print("Match {} conculed: player {} ({}) wins after {} ticks".format(self.match_num, winner, self.players[winner].name, self.board.time))
            self.wins[winner] += 1
            self._new_game()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_feature_count = get_in_feature_count()

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(in_feature_count, 512)
        self.bn1 = nn.LayerNorm(512)
        self.layer2 = nn.Linear(512, 128)
        self.bn2 = nn.LayerNorm(128)
        self.layer3 = nn.Linear(128, 32)
        self.bn3 = nn.LayerNorm(32)
        self.layer4 = nn.Linear(32, 8)
        self.bn4 = nn.LayerNorm(8)
        self.layer5 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.bn1(F.relu(self.layer1(x)))
        x = self.bn2(F.relu(self.layer2(x)))
        x = self.bn3(F.relu(self.layer3(x)))
        x = self.bn4(F.relu(self.layer4(x)))
        x = self.layer5(x)
        return x

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "terminal"))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, min(len(self.memory), batch_size))

    def __len__(self):
        return len(self.memory)

MEM_CAP = 32 * 1024
BATCH_SIZE = 1024
GAMMA = 0.995

losses = []

class Bot:
    def __init__(self, slot, model_filepath, learn):
        self.name = "bot"
        self.slot = slot
        self.model_filepath = model_filepath
        self.learn = learn
        self.train_net = DQN().to(device)
        #self.target_net = DQN().to(device)
        self.criterion = nn.MSELoss()
        #self.criterion = nn.CrossEntropyLoss()

        if os.path.isfile(self.model_filepath):
            net_state = torch.load(self.model_filepath)
            self.train_net.load_state_dict(net_state)
        else:
            net_state = self.train_net.state_dict()
        self.train_net.train(True)
        self._update_play_net(net_state)

        if self.learn:
            #self.optimizer = optim.SGD(self.train_net.parameters(), lr=0.01, momentum=0.9)
            self.optimizer = optim.RMSprop(self.train_net.parameters())
            #self.optimizer = optim.Adam(self.train_net.parameters(), lr=1e-6)
            self.memory = ReplayMemory(MEM_CAP)

    def _update_play_net(self, net_state):
        #self.target_net.load_state_dict(net_state)
        #self.target_net.eval()
        pass

    def new_game(self, board, match_num):
        self.board = board

        if self.learn and match_num % 10 == 0:
            net_state = self.train_net.state_dict()
            torch.save(net_state, self.model_filepath)
            self._update_play_net(net_state)
            #plt.plot(losses)
            #plt.savefig("loss-gravipong.png")

        if self.learn:
            self.eps_threshold = random.random() / (1.0003**match_num)
        else:
            self.eps_threshold = 0

    def __call__(self):
        self.last_state, _ = self.board.state(self.slot)
        self.last_action = self._select_action(self.last_state)
        return self.last_action.item()

    def reward(self, reward):
        if self.learn:
            # if reward > 0:
            #     reward *= 10
            reward = torch.tensor([reward], device=device)
            curr_state, terminal = self.board.state(self.slot)
            self.memory.push(self.last_state, self.last_action, curr_state, reward, terminal)
            self._optimize_model()

    def _select_action(self, state):
        sample = random.random()
        #eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * time / EPS_DECAY)
        if sample > self.eps_threshold:
            with torch.no_grad():
                noise = random.random()
                policies = self.train_net(state)
                policies[0] += 1e-6 * noise
                policies[1] += 1e-6 * (1 - noise)
                action = policies.max(0)[1].view(1, 1)
                if self.board.time % 25 == 0:
                    print("Q(s) policy: {} <- {}".format(action.item(), policies))
                return action
        else:
            return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)

    def _optimize_model(self):
        if (len(self.memory) < 3):
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch         = torch.stack(batch.state)
        action_batch        = torch.cat(batch.action)
        next_state_batch    = torch.stack(batch.next_state)
        reward_batch        = torch.cat(batch.reward)
        terminal_batch      = torch.tensor(batch.terminal).to(device)

        output_next_batch = self.train_net(next_state_batch)
        max_output_next_batch = output_next_batch.max(dim=1)

        y_batch_tuples = tuple(
            reward_batch[i] if terminal_batch[i] else reward_batch[i] + GAMMA * max_output_next_batch[0][i]
                for i in range(len(transitions)))
        y_batch = torch.stack(y_batch_tuples)

        Q_s = self.train_net(state_batch)
        state_action_values = torch.gather(Q_s, 1, action_batch.view(-1, 1))

        self.optimizer.zero_grad()

        y_batch = y_batch.view(-1, 1).detach()

        loss = self.criterion(state_action_values, y_batch)
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        #losses.append(loss.detach())

if __name__=="__main__":
    torch.set_printoptions(precision=2, linewidth=160, sci_mode=False)

    try:
        optlist, args = getopt.gnu_getopt(sys.argv[1:], "", ["no-disp", "no-vsync"])
    except getopt.GetoptError as err:
        print("error:" + str(err))
        sys.exit(2)

    if len(args) == 0:
        player_0 = "h"
        player_1 = "e"
    elif len(args) == 1:
        player_0 = args[0]
        player_1 = "e"
    elif len(args) == 2:
        player_0 = args[0]
        player_1 = args[1]
    else:
        print("error: there may be at most 2 players")
        sys.exit(2)

    display = not ("--no-disp", "") in optlist
    vsync = not ("--no-vsync", "") in optlist

    game = Game([player_0, player_1], display, vsync)
    game.run()
