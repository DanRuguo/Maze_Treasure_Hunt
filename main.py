import sys
import matplotlib.pyplot as plt
import pandas as pd
import time

if "../" not in sys.path:
    sys.path.append("../")
from maze import Maze
from RL_brain import QLearningTable, SarsaTable
import numpy as np

METHOD = "SARSA"
# METHOD = "Q-Learning"

rewards_per_episode = []

def get_action(q_table, state):
    # 选择最优行为
    state_action = q_table.loc[state, :]

    # 因为一个状态下最优行为可能会有多个,所以在碰到这种情况时,需要随机选择一个行为进行
    state_action_max = state_action.max()

    idxs = []

    for max_item in range(len(state_action)):
        if state_action[max_item] == state_action_max:
            idxs.append(max_item)

    sorted(idxs)
    return tuple(idxs)


def get_policy(q_table, rows=5, cols=5, pixels=40, orign=20):
    policy = []

    for i in range(rows):
        for j in range(cols):
            # 求出每个各自的状态
            item_center_x, item_center_y = (j * pixels + orign), (i * pixels + orign)
            item_state = [item_center_x - 15.0, item_center_y - 15.0, item_center_x + 15.0, item_center_y + 15.0]

            # 如果当前状态为各终止状态,则值为-1
            if item_state in [env.canvas.coords(env.hell1), env.canvas.coords(env.hell2),
                              env.canvas.coords(env.hell3), env.canvas.coords(env.hell4),
                              env.canvas.coords(env.hell5), env.canvas.coords(env.hell6),
                              env.canvas.coords(env.hell7), env.canvas.coords(env.oval)]:
                policy.append(-1)
                continue

            if str(item_state) not in q_table.index:
                policy.append((0, 1, 2, 3))
                continue

            # 选择最优行为
            item_action_max = get_action(q_table, str(item_state))

            policy.append(item_action_max)

    return policy

def policy_to_csv(policy_result, filename):
    """
    将策略结果保存为 CSV 文件
    """
    df = pd.DataFrame(policy_result)
    df.to_csv(filename, index=False, header=False)

def update():
    num_episodes = 100
    global rewards_per_episode
    rewards_per_episode = []

    for episode in range(num_episodes):
        # 初始化状态
        observation = env.reset()
        episode_reward = 0

        tmp_policy = {}

        while True:
            # 渲染当前环境
            env.render()

            # 基于当前状态选择行为
            action = RL.choose_action(str(observation))

            state_item = tuple(observation)

            tmp_policy[state_item] = action

            # 采取行为获得下一个状态和回报,及是否终止
            observation_, reward, done, oval_flag = env.step(action)

            episode_reward += reward

            if METHOD == "SARSA":
                # 基于下一个状态选择行为
                action_ = RL.choose_action(str(observation_))

                # 基于变化 (s, a, r, s, a) 使用 Sarsa 进行 Q 的更新
                RL.learn(str(observation), action, reward, str(observation_), action_)
            elif METHOD == "Q-Learning":
                # 根据当前的变化开始更新 Q
                RL.learn(str(observation), action, reward, str(observation_))

            # 改变状态和行为
            observation = observation_

            # 如果为终止状态,结束当前的局数
            if done:
                break

        rewards_per_episode.append(episode_reward)

    print('游戏结束')
    # 开始输出最终的 Q 表
    q_table_result = RL.q_table
    # 使用 Q 表输出各状态的最优策略
    policy = get_policy(q_table_result)
    print("最优策略为", end=":\n")
    print(policy)
    print("迷宫格式为", end=":\n")
    # 迷宫大小是 5x5
    maze_size = 5

    # 初始化一个与迷宫大小相同的策略网格，填充 None
    policy_result = [[None for _ in range(maze_size)] for _ in range(maze_size)]

    # 填充策略网格
    index = 0  # 用于在 policy 列表中追踪位置
    for i in range(maze_size):
        for j in range(maze_size):
            # 从 policy 列表获取策略并放入网格
            policy_result[i][j] = policy[index]
            index += 1  # 移动到下一个元素
    print(policy_result)
    print("根据求出的最优策略画出方向(见迷宫)", end=":\n")
    env.render_by_policy_new(policy_result)
    
    # 动态命名文件并保存图像
    filename = f"{METHOD.lower()}.png"
    env.save_canvas(filename)
    # # 保存策略为 CSV 文件
    policy_csv_filename = f"{METHOD.lower()}_policy_result.csv"
    policy_to_csv(policy_result, policy_csv_filename)
    # 这会使 Tkinter 等待 5000 毫秒，然后调用 env.destroy() 方法
    env.after(5000, env.destroy)


def plot_rewards(sarsa_rewards, q_learning_rewards):
    plt.plot(sarsa_rewards, label="SARSA")
    plt.plot(q_learning_rewards, label="Q-Learning")
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.savefig("sarsa_vs_qlearning_rewards.png")  # 保存图像
    plt.show()


if __name__ == "__main__":
    # SARSA 收敛曲线
    print("下面执行SARSA方法......")
    METHOD = "SARSA"
    start_time = time.time()
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()
    sarsa_rewards = rewards_per_episode
    sarsa_time = time.time() - start_time
    print(f"SARSA 总运行时间: {sarsa_time:.2f} 秒")

    # Q-Learning 收敛曲线
    print("下面执行Q-Learning方法......")
    METHOD = "Q-Learning"
    start_time = time.time()
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()
    q_learning_rewards = rewards_per_episode
    q_learning_time = time.time() - start_time
    print(f"Q-Learning 总运行时间: {q_learning_time:.2f} 秒")

    # 绘制收敛曲线
    plot_rewards(sarsa_rewards, q_learning_rewards)
