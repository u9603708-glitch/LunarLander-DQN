"""
DQN for LunarLander-v3
======================

Adapted from Chapter 3 of "Deep Reinforcement Learning in Action"
(Listing 3.7: DQN with experience replay + target network).

What is reused from Chapter 3 (almost line for line):
    * A small fully connected Q-network:  state -> Q(s, a) for every action
    * Epsilon-greedy action selection
    * Experience replay stored in a `deque`, sampled in random minibatches
    * A second "target" network (model2) that is synchronised every
      `sync_freq` steps and used to compute the TD target
    * The TD target  Y = r + gamma * (1 - done) * max_a' Q_target(s', a')
    * MSE loss between Q(s, a) and Y, optimised with Adam

What had to change for LunarLander (the "minor adjustments"):
    1. Environment API: Gridworld -> gymnasium.make("LunarLander-v3").
       The state is already a numeric vector of 8 floats, so no
       `render_np().reshape(1, 64)` and no added noise are needed.
    2. Network size: input 8 (state dim), output 4 (number of actions).
    3. gamma 0.9 -> 0.99: most of the reward is shaping given every step,
       and the +100 landing bonus arrives in the last few dozen steps.
       With gamma = 0.9 a reward 20 steps ahead keeps only ~12% of its
       value; with 0.99 it keeps ~82%, so the landing bonus still matters.
    4. Replay memory 1,000 -> 100,000 transitions. One LunarLander episode
       alone can fill 1,000 slots, so a tiny buffer would contain only
       highly correlated samples from the latest episode.
    5. Epsilon decays from 1.0 to 0.01 instead of staying at 0.3, so the
       agent explores a lot early and exploits what it has learned later.
    6. Terminal vs. truncated: gymnasium returns two flags. Only a real
       terminal state (crash / landed) cuts the bootstrap; a time-limit
       truncation does NOT (same idea as the Chapter 3 comment "this is
       not an absorbing state").
    7. Gradient-norm clipping, to stop rare huge TD errors (crash = -100)
       from blowing up the weights.

Run:
    pip install "gymnasium[box2d]" torch matplotlib
    python dqn_lunarlander.py                 # train + evaluate
    python dqn_lunarlander.py --episodes 800  # longer training
    python dqn_lunarlander.py --render        # watch the trained agent
"""

import argparse
import copy
import random
from collections import deque

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")  # save figures to file; works without a display
import numpy as np
import torch
from matplotlib import pyplot as plt


# ----------------------------------------------------------------------------
# 0. Reproducibility
# ----------------------------------------------------------------------------
def set_seed(seed):
    """Fix every random number generator we use so a run can be repeated.

    DQN results vary a lot between seeds, so when we compare two settings
    (e.g. two learning rates) we must use the same seed(s) for both.
    """
    random.seed(seed)        # used by random.random() / random.sample()
    np.random.seed(seed)     # used by np.random.randint() for random actions
    torch.manual_seed(seed)  # used for the initial network weights


# ----------------------------------------------------------------------------
# 1. Hyper-parameters (Chapter 3 values in comments for comparison)
# ----------------------------------------------------------------------------
L1 = 8     # input size = state dimension of LunarLander       (Ch3: 64)
L2 = 150   # first hidden layer  - kept identical to Chapter 3
L3 = 100   # second hidden layer - kept identical to Chapter 3
L4 = 4     # output size = number of discrete actions          (Ch3: 4)
#            0: do nothing, 1: fire left engine,
#            2: fire main engine, 3: fire right engine

LEARNING_RATE = 5e-4   # Adam step size                        (Ch3: 1e-3)
GAMMA = 0.99           # discount factor                       (Ch3: 0.9)
MEM_SIZE = 100_000     # replay memory capacity                (Ch3: 1000)
BATCH_SIZE = 64        # minibatch size                        (Ch3: 200)
SYNC_FREQ = 1000       # copy model -> model2 every N steps    (Ch3: 500)
TRAIN_EVERY = 4        # do one gradient step every N env steps (Ch3: 1)
LEARN_START = 1000     # collect this many transitions before learning
EPS_START = 1.0        # initial exploration rate              (Ch3: 0.3 fixed)
EPS_END = 0.01         # minimum exploration rate
EPS_DECAY = 0.995      # multiply epsilon by this after every episode
MAX_GRAD_NORM = 10.0   # gradient clipping threshold (new, not in Ch3)
SOLVED_SCORE = 200.0   # Gymnasium defines "solved" as avg reward >= 200


# ----------------------------------------------------------------------------
# 2. The Q-network (same architecture as Chapter 3, new input size)
# ----------------------------------------------------------------------------
def build_model():
    """Return a 3-layer MLP that maps a state (8 numbers) to 4 Q-values.

    Q(s, a) estimates the total discounted reward we will collect if we take
    action `a` in state `s` and then act greedily afterwards. The network
    outputs all 4 Q-values at once, so choosing the best action is just an
    argmax over the output vector.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(L1, L2),
        torch.nn.ReLU(),
        torch.nn.Linear(L2, L3),
        torch.nn.ReLU(),
        torch.nn.Linear(L3, L4),  # linear output: Q-values can be any sign
    )


# ----------------------------------------------------------------------------
# 3. Helpers
# ----------------------------------------------------------------------------
def to_tensor(state):
    """numpy array (8,) -> float tensor of shape (1, 8).

    The leading batch dimension of 1 mirrors `.reshape(1, 64)` in Chapter 3,
    which lets us later `torch.cat` many states into one minibatch.
    """
    return torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0)


def select_action(model, state1, epsilon):
    """Epsilon-greedy policy (identical logic to Chapter 3, comment #F).

    With probability epsilon we pick a random action (explore); otherwise
    we pick the action with the highest predicted Q-value (exploit).
    """
    if random.random() < epsilon:
        return np.random.randint(0, L4)
    with torch.no_grad():               # no gradients needed to act
        qval = model(state1)
    return int(torch.argmax(qval, dim=1).item())


def running_mean(x, N=50):
    """Moving average used to smooth the noisy loss / reward curves.

    Same purpose as `running_mean` in Chapter 3, implemented with
    np.convolve for speed.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < N:
        return x
    return np.convolve(x, np.ones(N) / N, mode="valid")


# ----------------------------------------------------------------------------
# 4. Training loop (structure of Chapter 3, Listing 3.7)
# ----------------------------------------------------------------------------
def train(episodes, seed):
    env = gym.make("LunarLander-v3")

    model = build_model()                 # online network: trained every step
    model2 = copy.deepcopy(model)         # target network: frozen copy (Ch3 #A)
    model2.load_state_dict(model.state_dict())

    loss_fn = torch.nn.MSELoss()          # same loss as Chapter 3
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Each element is one transition (s, a, r, s', terminated).
    # When the deque is full, the oldest transition is dropped automatically.
    replay = deque(maxlen=MEM_SIZE)

    epsilon = EPS_START
    losses = []          # one entry per gradient step
    episode_rewards = [] # total reward of each training episode
    j = 0                # global step counter (used for sync / train timing)

    for i in range(episodes):
        # A new episode: reset the lander to a random starting condition.
        # Seeding only the first reset makes the whole run reproducible
        # while still giving every episode a different start.
        state, _ = env.reset(seed=seed if i == 0 else None)
        state1 = to_tensor(state)
        done = False
        total_reward = 0.0

        while not done:
            j += 1

            # --- (a) act in the environment --------------------------------
            action_ = select_action(model, state1, epsilon)
            state, reward, terminated, truncated, _ = env.step(action_)
            # terminated: lander crashed, landed and came to rest, or flew
            #             off-screen -> a true absorbing state, V(s') = 0.
            # truncated:  hit the 1000-step time limit -> NOT absorbing; the
            #             future still has value, so we keep bootstrapping.
            done = terminated or truncated
            state2 = to_tensor(state)
            total_reward += reward

            # --- (b) store experience (Ch3 #G, #H) --------------------------
            # We store `terminated`, not `done`, so time-limit truncation does
            # not wrongly tell the network "nothing happens after this".
            replay.append((state1, action_, reward, state2, terminated))
            state1 = state2

            # --- (c) learn from a random minibatch (Ch3 #I .. #N) ----------
            if len(replay) >= LEARN_START and j % TRAIN_EVERY == 0:
                minibatch = random.sample(replay, BATCH_SIZE)

                # Split the list of tuples into 5 batched tensors.
                state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.tensor(
                    [r for (s1, a, r, s2, d) in minibatch], dtype=torch.float32)
                state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.tensor(
                    [d for (s1, a, r, s2, d) in minibatch], dtype=torch.float32)

                # Q(s, .) from the online network - gradients flow through it.
                Q1 = model(state1_batch)
                # Q(s', .) from the TARGET network - no gradients. Using a
                # slowly changing copy keeps the target from "chasing itself",
                # which is the main source of instability in plain DQN.
                with torch.no_grad():
                    Q2 = model2(state2_batch)

                # Bellman target. (1 - done) zeroes the future term for
                # terminal transitions, exactly as in Chapter 3 comment #N.
                Y = reward_batch + GAMMA * (1 - done_batch) * torch.max(Q2, dim=1)[0]

                # Pick out Q(s, a) only for the action that was actually taken.
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(1)).squeeze(1)

                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                # Clip the gradient norm: a crash gives -100 reward, which can
                # produce a very large TD error and one destructive update.
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                losses.append(loss.item())

            # --- (d) periodically refresh the target network (Ch3 #C) -------
            if j % SYNC_FREQ == 0:
                model2.load_state_dict(model.state_dict())

        # Decay exploration once per episode, but never below EPS_END.
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        episode_rewards.append(total_reward)

        # Progress report every 20 episodes.
        if (i + 1) % 20 == 0:
            avg_r = np.mean(episode_rewards[-100:])
            avg_l = np.mean(losses[-1000:]) if losses else float("nan")
            print(f"Episode {i + 1:4d} | avg reward (last 100) {avg_r:8.2f} "
                  f"| avg loss (last 1000 updates) {avg_l:8.3f} "
                  f"| epsilon {epsilon:.3f}")

    env.close()
    return model, np.array(losses), np.array(episode_rewards)


# ----------------------------------------------------------------------------
# 5. Evaluation (counterpart of `test_model` in Chapter 3)
# ----------------------------------------------------------------------------
def test_model(model, n_games=100, render=False, seed=1000):
    """Play `n_games` episodes with the greedy policy (epsilon = 0).

    In Chapter 3 we counted wins; here the natural metric is the episode
    return, and an average >= 200 over 100 games means the task is solved.
    """
    env = gym.make("LunarLander-v3", render_mode="human" if render else None)
    scores = []
    for k in range(n_games):
        state, _ = env.reset(seed=seed + k)  # unseen start states
        done, total = False, 0.0
        while not done:
            action = select_action(model, to_tensor(state), epsilon=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        scores.append(total)
    env.close()
    scores = np.array(scores)
    print(f"Test over {n_games} games: mean {scores.mean():.2f}, "
          f"std {scores.std():.2f}, landed (>=200): {(scores >= 200).mean() * 100:.0f}%")
    return scores


# ----------------------------------------------------------------------------
# 6. Plots
# ----------------------------------------------------------------------------
def plot_results(losses, rewards, path="dqn_lunarlander_results.png"):
    """Draw the training loss (like Chapter 3) and the episode reward."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(running_mean(losses, 500))
    ax[0].set_xlabel("Gradient updates")
    ax[0].set_ylabel("Running average MSE loss (500)")
    ax[0].set_title("Training loss")

    ax[1].plot(rewards, alpha=0.3, label="episode reward")
    ax[1].plot(np.arange(49, len(rewards)), running_mean(rewards, 50),
               label="running mean (50)")
    ax[1].axhline(SOLVED_SCORE, color="gray", ls="--", label="solved = 200")
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("Total reward")
    ax[1].set_title("Episode reward")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved figure to {path}")


# ----------------------------------------------------------------------------
# 7. Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true",
                        help="load saved weights and watch 3 games")
    args = parser.parse_args()

    if args.render:
        model = build_model()
        model.load_state_dict(torch.load("dqn_lunarlander.pt"))
        test_model(model, n_games=3, render=True)
    else:
        set_seed(args.seed)
        model, losses, rewards = train(args.episodes, args.seed)
        torch.save(model.state_dict(), "dqn_lunarlander.pt")

        # A note on the "final loss": in DQN the loss is a TD error against a
        # moving target, not a fixed supervised error. It typically RISES
        # while the agent improves (Q-values and rewards get larger), then
        # settles. The reward curve is the real measure of success.
        print(f"Final loss (mean of last 1000 updates): {losses[-1000:].mean():.4f}")
        print(f"Final loss (last update):              {losses[-1]:.4f}")

        test_model(model, n_games=100)
        plot_results(losses, rewards)
