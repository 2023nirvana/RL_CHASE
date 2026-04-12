import numpy as np  
import random  
from collections import defaultdict  
  
LOW_LEVEL_ACTIONS = {  
    0: "hover",  
    1: "surge_forward",  
    2: "surge_backward",  
    3: "sway_left",  
    4: "sway_right",  
    5: "heave_up",  
    6: "heave_down",  
    7: "yaw_left",  
    8: "yaw_right",  
    9: "pitch_up",  
    10: "pitch_down",  
    11: "roll_left",  
    12: "roll_right",  
}  
  
class SimplifiedLowLevelEnv:  
    def __init__(self):  
        self.dt = 0.1  
        self.max_steps = 200  
        self.position_bins = np.linspace(-5, 5, 21)  
        self.orientation_bins = np.linspace(-0.5, 0.5, 21)  
        self.velocity_bins = np.linspace(-1.0, 1.0, 21)  
        self.target = np.zeros(12)  
        self.reset()  
  
    def reset(self):  
        self.state = np.zeros(12)  
        self.steps = 0  
        return self._discretize_state(self.state)  
  
    def step(self, action):  
        self.steps += 1  
        self.state = self._apply_action(self.state, action)  
        self.state = self._update_state(self.state)  
        obs = self._discretize_state(self.state)  
        reward = self._compute_reward(self.state)  
        done = self._check_done(self.state)  
        return obs, reward, done, {}  
  
    def _apply_action(self, state, action):  
        next_state = state.copy()  
        linear_delta = 0.05  
        ang_delta = 0.02  
  
        if action == 1:  
            next_state[6] += linear_delta  
        elif action == 2:  
            next_state[6] -= linear_delta  
        elif action == 3:  
            next_state[7] -= linear_delta  
        elif action == 4:  
            next_state[7] += linear_delta  
        elif action == 5:  
            next_state[8] += linear_delta  
        elif action == 6:  
            next_state[8] -= linear_delta  
        elif action == 7:  
            next_state[11] -= ang_delta  
        elif action == 8:  
            next_state[11] += ang_delta  
        elif action == 9:  
            next_state[10] += ang_delta  
        elif action == 10:  
            next_state[10] -= ang_delta  
        elif action == 11:  
            next_state[9] -= ang_delta  
        elif action == 12:  
            next_state[9] += ang_delta  
  
        return next_state  
  
    def _update_state(self, state):  
        next_state = state.copy()  
        next_state[0:3] += state[6:9] * self.dt  
        next_state[3:6] += state[9:12] * self.dt  
        next_state[6:9] *= 0.95  
        next_state[9:12] *= 0.95  
        return next_state  
  
    def _discretize_state(self, state):  
        idx = []  
        for i in range(3):  
            idx.append(int(np.digitize(state[i], self.position_bins) - 1))  
        for i in range(3, 6):  
            idx.append(int(np.digitize(state[i], self.orientation_bins) - 1))  
        for i in range(6, 9):  
            idx.append(int(np.digitize(state[i], self.velocity_bins) - 1))  
        for i in range(9, 12):  
            idx.append(int(np.digitize(state[i], self.velocity_bins) - 1))  
        return tuple(np.clip(idx, 0, len(self.position_bins) - 2))  
  
    def _compute_reward(self, state):  
        pos_err = np.linalg.norm(state[:3])  
        ori_err = np.linalg.norm(state[3:6])  
        vel_pen = np.linalg.norm(state[6:12])  
        reward = -2.0 * pos_err - 1.5 * ori_err - 0.2 * vel_pen  
        if self._check_done(state):  
            reward += 10.0  
        return reward  
  
    def _check_done(self, state):  
        pos_err = np.linalg.norm(state[:3])  
        ori_err = np.linalg.norm(state[3:6])  
        if pos_err < 0.2 and ori_err < 0.1:  
            return True  
        if self.steps >= self.max_steps:  
            return True  
        return False  
  
  
def q_learning(env, episodes=5000, alpha=0.2, gamma=0.99, eps=0.1):  
    Q = defaultdict(float)  
    returns = []  
    for ep in range(episodes):  
        state = env.reset()  
        total_reward = 0  
        for t in range(200):  
            if random.random() < eps:  
                action = random.randint(0, 12)  
            else:  
                q_vals = [Q[(state, a)] for a in range(13)]  
                action = int(np.argmax(q_vals))  
            next_state, reward, done, _ = env.step(action)  
            total_reward += reward  
            best_next = max(Q[(next_state, a)] for a in range(13))  
            Q[(state, action)] += alpha * (reward + gamma * best_next - Q[(state, action)])  
            state = next_state  
            if done:  
                break  
        returns.append(total_reward)  
        if (ep + 1) % 100 == 0:  
            print(f"Episode {ep+1}, AvgReturn {np.mean(returns[-100:]):.2f}")  
    return Q, returns  
  
if __name__ == "__main__":  
    env = SimplifiedLowLevelEnv()  
    Q, returns = q_learning(env)  
    print("最后100集平均回报:", np.mean(returns[-100:]))  
