import numpy as np
import matplotlib
matplotlib.use('Agg') # 確保在無介面環境也能繪圖
import matplotlib.pyplot as plt
from flask import Flask, render_template, jsonify, request
import os

app = Flask(__name__)

# --- 環境設定 ---
WIDTH, HEIGHT = 12, 4
START, GOAL = (3, 0), (3, 11)
CLIFF = [(3, i) for i in range(1, 11)]
ACTIONS = [0, 1, 2, 3] # 0:上, 1:下, 2:左, 3:右
ACTION_EFFECTS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

class CliffWalkingAgent:
    def __init__(self, method="q_learning", alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((HEIGHT, WIDTH, 4))
        self.method = method
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(ACTIONS)
        return int(np.argmax(self.q_table[state[0], state[1]]))

    def update(self, s, a, r, s_next, a_next=None):
        current_q = self.q_table[s[0], s[1], a]
        if self.method == "q_learning":
            # Off-policy: Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
            max_next_q = np.max(self.q_table[s_next[0], s_next[1]])
            td_target = r + self.gamma * max_next_q
        else:
            # On-policy: Q(s,a) = Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
            td_target = r + self.gamma * self.q_table[s_next[0], s_next[1], a_next]
        self.q_table[s[0], s[1], a] += self.alpha * (td_target - current_q)

def save_homework_images(q_agent, q_rewards, s_agent, s_rewards):
    # 1. 繪製 result_sample.jpg
    plt.figure(figsize=(10, 5))
    def smooth(data, window=10):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    plt.plot(smooth(q_rewards), label='Q-learning', color='red')
    plt.plot(smooth(s_rewards), label='SARSA', color='teal')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards per episode')
    plt.title('SARSA vs Q-learning (Cliff Walking)')
    plt.legend()
    plt.grid(True)
    plt.ylim(-100, 0)
    plt.savefig('result_sample.jpg', dpi=300)
    plt.close()

    # 2. 繪製 cliff.jpg
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    arrows_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    def draw_ax(ax, agent, title):
        ax.set_title(title)
        policy = np.argmax(agent.q_table, axis=2)
        for r in range(HEIGHT):
            for c in range(WIDTH):
                if r == 3 and 1 <= c <= 10:
                    ax.text(c, r, 'CLIFF', ha='center', va='center', color='white', weight='bold', fontsize=8)
                    ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='#333'))
                elif (r, c) == START: ax.text(c, r, 'S', ha='center', va='center', color='blue', weight='bold')
                elif (r, c) == GOAL: ax.text(c, r, 'G', ha='center', va='center', color='red', weight='bold')
                else: ax.text(c, r, arrows_map[policy[r,c]], ha='center', va='center', fontsize=14)
        ax.set_xticks(np.arange(WIDTH))
        ax.set_yticks(np.arange(HEIGHT))
        ax.grid(True)

    draw_ax(ax1, q_agent, "Q-learning Policy")
    draw_ax(ax2, s_agent, "SARSA Policy")
    plt.tight_layout()
    plt.savefig('cliff.jpg', dpi=300)
    plt.close()

def run_simulation(method, episodes):
    agent = CliffWalkingAgent(method=method)
    rewards_history = []
    for _ in range(episodes):
        s = START
        total_r = 0
        a = agent.choose_action(s)
        while s != GOAL:
            eff = ACTION_EFFECTS[a]
            s_next = (max(0, min(HEIGHT-1, s[0] + eff[0])), max(0, min(WIDTH-1, s[1] + eff[1])))
            reward = -100 if s_next in CLIFF else -1
            if s_next in CLIFF: s_next = START
            a_next = agent.choose_action(s_next)
            agent.update(s, a, reward, s_next, a_next)
            s, a, total_r = s_next, a_next, total_r + reward
        rewards_history.append(total_r)
    return agent, rewards_history

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    episodes = request.json.get('episodes', 500)
    q_agent, q_rewards = run_simulation("q_learning", episodes)
    s_agent, s_rewards = run_simulation("sarsa", episodes)
    save_homework_images(q_agent, q_rewards, s_agent, s_rewards)
    return jsonify({
        "q_learning": {"policy": np.argmax(q_agent.q_table, axis=2).tolist(), "rewards": q_rewards},
        "sarsa": {"policy": np.argmax(s_agent.q_table, axis=2).tolist(), "rewards": s_rewards}
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)