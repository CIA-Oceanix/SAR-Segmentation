import numpy as np
import random
from collections import deque
import skimage
from datetime import datetime

from Rignak_DeepLearning.ReinforcementLearning.plot_learning import Plot

ARGS = {
    'final_epsilon': 0.05,
    'ini_epsilon': 1.0,
    'exploration': 50000,
    'batch_size': 128,
    'gamma': 0.99,
    'replay': 20000,
    'observation': 2000,
    'observation_delay': 1000,
    'mode': "Train",
    "frame_skip": 1,
    'fps': 30}


def format_state(state, shape):
    state = skimage.color.rgb2gray(state)
    if shape != state[:2].shape:
        state = skimage.transform.resize(state, shape)
    return state[:,:,0]


def apply_update_rule(D, batch_size, model, gamma):
    batch = random.sample(D, batch_size)
    previous_state, current_state, action, reward_t, terminal = zip(*batch)
    previous_state = np.concatenate(previous_state)
    current_state = np.concatenate(current_state)

    targets = model.predict(previous_state)

    Q_sa = model.predict(current_state)
    maxQ_sa = np.max(Q_sa, axis=1)
    targets[range(batch_size), action] = reward_t + gamma * maxQ_sa * np.invert(terminal)
    return targets, previous_state, maxQ_sa


def train(library, model, shape, command_number, GameState, args=ARGS):
    model.plot = Plot()
    history = {'cumulative_reward': 0, 'Qsa_maxmean': 0, 'loss_mean': 0, 'exp': [0, 0], 'frames': [], "actions": []}

    game = GameState()

    D = deque()
    do_nothing = np.zeros(command_number)
    do_nothing[0] = 1
    state0, reward, terminal = game.turn(do_nothing, fps=None)
    state0 = format_state(state0, shape)
    state0 = np.stack((state0, state0, state0, state0), axis=2)
    state0 = np.expand_dims(state0, axis=0)


    current_epsilon = args['ini_epsilon']

    if args['mode'] == 'Run':
        fps = args['fps']
    else:
        fps = None

    t = 0
    begin = None
    while True:
        library.Inputs.manual_input()
        if t < args['observation'] and current_epsilon > random.random() and args['mode'] != 'Run':
            action = np.random.randint(0, command_number, 1)[0]
            action_array = do_nothing.copy()
        else:
            action_array = model.predict(state0)[0]
            print(action_array)
            history['actions'].append(action_array)
            action = np.argmax(action_array[1:]) + 1
            history['exp'] = [history['exp'][0] + action_array[action], history['exp'][1] + 1]
            action_array = model.predict(state0)[0]
        action_array[action] = 1
        action_array[~action] = 0

        if current_epsilon > args['final_epsilon'] and t > args['observation']:
            current_epsilon -= current_epsilon

        state1, reward, terminal = game.turn(action_array, fps=fps)
        if terminal:
            history['frames'].append(terminal)
            terminal = True
            game = GameState()

        history['cumulative_reward'] += reward

        state1 = format_state(state1, shape)
        state1 = np.expand_dims(state1, axis=0)
        state1 = np.expand_dims(state1, axis=-1)
        state1 = np.append(state1, state0[:, :, :, :3], axis=3)
        D.append((state0, state1, action, reward, terminal))

        if len(D) > args['replay']:
            D.popleft()

        if t > args['observation'] and t % args['frame_skip'] == 0:
            targets, input_, maxQ_sa = apply_update_rule(D, args['batch_size'], model, args['gamma'])
            history['loss_mean'] += model.train_on_batch(input_, targets)[0]
            history['Qsa_maxmean'] += max(maxQ_sa)

        if t % args['observation_delay'] == 0 and t > args['observation']:
            if begin is None:
                begin = datetime.now()
                history = {'cumulative_reward': 0, 'Qsa_maxmean': 0, 'loss_mean': 0, 'exp': [0, 0], 'frames': [],
                           'actions': []}
                continue
            history['cumulative_reward'] /= args['observation_delay']
            history['frames'] = np.mean(history['frames'])
            history['actions'] = np.mean(np.array(history['actions']), axis=0)
            history['Qsa_maxmean'] /= args['observation_delay']
            history['exp'] = history['exp'][0] / history['exp'][1]
            print(f'{t} - {round(history["cumulative_reward"], 5)} - {history["actions"]}')

            model.plot.Plot(history, t)

            history = {'cumulative_reward': 0, 'Qsa_maxmean': 0, 'loss_mean': 0, 'exp': [0, 0], 'frames': [],
                       'actions': []}

        state0 = state1
        t += 1
