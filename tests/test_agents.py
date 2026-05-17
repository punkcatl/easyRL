import numpy as np
import torch
from algorithms.q_learning.agent import QLearningAgent


def test_qlearning_agent_init():
    agent = QLearningAgent(n_states=48, n_actions=4, lr=0.1, gamma=0.99, epsilon=0.1)
    assert agent.q_table.shape == (48, 4)
    assert np.all(agent.q_table == 0)


def test_qlearning_agent_select_action():
    agent = QLearningAgent(n_states=48, n_actions=4, lr=0.1, gamma=0.99, epsilon=0.0)
    agent.q_table[0, 2] = 10.0
    action = agent.select_action(0)
    assert action == 2


def test_qlearning_agent_update():
    agent = QLearningAgent(n_states=48, n_actions=4, lr=0.5, gamma=0.9, epsilon=0.1)
    agent.update(state=0, action=1, reward=1.0, next_state=1, done=False)
    assert agent.q_table[0, 1] > 0


from algorithms.dqn.agent import DQNAgent


def test_dqn_agent_init():
    agent = DQNAgent(state_dim=4, action_dim=2, lr=1e-3, gamma=0.99, epsilon=1.0, buffer_size=1000, batch_size=32)
    assert agent is not None


def test_dqn_agent_select_action():
    agent = DQNAgent(state_dim=4, action_dim=2, lr=1e-3, gamma=0.99, epsilon=0.0, buffer_size=1000, batch_size=32)
    state = np.zeros(4)
    action = agent.select_action(state)
    assert action in [0, 1]


def test_dqn_agent_store_and_learn():
    agent = DQNAgent(state_dim=4, action_dim=2, lr=1e-3, gamma=0.99, epsilon=1.0, buffer_size=100, batch_size=4)
    for i in range(10):
        state = np.random.randn(4)
        action = agent.select_action(state)
        next_state = np.random.randn(4)
        agent.store_transition(state, action, 1.0, next_state, False)
    loss = agent.learn()
    assert loss is not None


from algorithms.policy_gradient.agent import REINFORCEAgent


def test_reinforce_agent_init():
    agent = REINFORCEAgent(state_dim=4, action_dim=2, lr=1e-3, gamma=0.99)
    assert agent is not None


def test_reinforce_agent_select_action():
    agent = REINFORCEAgent(state_dim=4, action_dim=2, lr=1e-3, gamma=0.99)
    state = np.zeros(4)
    action = agent.select_action(state)
    assert action in [0, 1]


def test_reinforce_agent_update():
    agent = REINFORCEAgent(state_dim=4, action_dim=2, lr=1e-3, gamma=0.99)
    state = np.zeros(4)
    for _ in range(5):
        action = agent.select_action(state)
        agent.store_reward(1.0)
    loss = agent.update()
    assert loss is not None


from algorithms.ppo.agent import PPOAgent


def test_ppo_agent_init():
    agent = PPOAgent(state_dim=25, action_dim=2, lr=3e-4, gamma=0.99, clip_eps=0.2, epochs=10, batch_size=64)
    assert agent is not None


def test_ppo_agent_select_action():
    agent = PPOAgent(state_dim=25, action_dim=2, lr=3e-4, gamma=0.99, clip_eps=0.2, epochs=10, batch_size=64)
    state = np.zeros(25)
    action, log_prob, value = agent.select_action(state)
    assert action.shape == (2,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)


def test_ppo_agent_update():
    agent = PPOAgent(state_dim=25, action_dim=2, lr=3e-4, gamma=0.99, clip_eps=0.2, epochs=2, batch_size=4)
    states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
    for _ in range(10):
        s = np.random.randn(25)
        a, lp, v = agent.select_action(s)
        states.append(s)
        actions.append(a)
        rewards.append(1.0)
        log_probs.append(lp)
        values.append(v)
        dones.append(False)
    agent.update(states, actions, rewards, log_probs, values, dones, next_value=0.0)


from algorithms.sac.agent import SACAgent


def test_sac_agent_init():
    agent = SACAgent(state_dim=3, action_dim=1, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, buffer_size=10000, batch_size=64)
    assert agent is not None


def test_sac_agent_select_action():
    agent = SACAgent(state_dim=3, action_dim=1, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, buffer_size=10000, batch_size=64)
    state = np.zeros(3)
    action = agent.select_action(state)
    assert action.shape == (1,)
    assert -1.0 <= action[0] <= 1.0


def test_sac_agent_store_and_learn():
    agent = SACAgent(state_dim=3, action_dim=1, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, buffer_size=100, batch_size=4)
    for _ in range(10):
        state = np.random.randn(3)
        action = agent.select_action(state)
        next_state = np.random.randn(3)
        agent.store_transition(state, action, 1.0, next_state, False)
    result = agent.learn()
    assert result is not None
