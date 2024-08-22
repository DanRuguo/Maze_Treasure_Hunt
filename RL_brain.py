import numpy as np
import pandas as pd


class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        # 在强化学习中，智能体（agent）需要在每个时间步从这个空间中选择一个动作来执行。
        self.actions = action_space
        # 用于控制智能体更新其学习（如 Q 值）时的速度。
        self.lr = learning_rate
        # 用于确定未来奖励的当前价值。值越接近于 1，智能体在做出决策时越看重未来的回报；值越低，智能体越倾向于优化即时回报。这有助于调节智能体的短期与长期目标之间的平衡。
        self.gamma = reward_decay
        # 设定了智能体在选择最佳已知动作（利用）与随机选择动作以探索未知的环境（探索）之间的平衡。值越高，智能体在初期进行更多的随机探索，有助于获得更多关于环境的信息，但可能会在短期内牺牲一些性能。
        self.epsilon = e_greedy
        # 用于存储智能体在环境中学习到的知识。在 Q 学习中，Q 表是一个状态-动作对应的表格，每个状态-动作对应一个 Q 值，表示智能体在这个状态下选择这个动作的价值。
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # 如果状态在当前的Q表中不存在，将当前状态加入Q表中
            # 创建一个新的 DataFrame 来表示新状态
            new_state_row = pd.DataFrame(
                [0] * len(self.actions),
                index=self.q_table.columns,
                columns=[state]
            ).T  # 转置以匹配行标签为状态，列标签为动作的格式
            # 使用 concat 方法添加到原来的 Q 表中
            self.q_table = pd.concat([self.q_table, new_state_row])

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # 从均匀分布的[0,1)中随机采样,当小于阈值时采用选择最优行为的方式,当大于阈值选择随机行为的方式,这样人为增加随机性是为了解决陷入局部最优
        if np.random.rand() < self.epsilon:
            # 选择最优行为
            state_action = self.q_table.loc[observation, :]
            # 因为一个状态下最优行为可能会有多个,所以在碰到这种情况时,需要随机选择一个行为进行
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            # 选择随机行为
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass


# 离轨策略Q-learning
class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, e_decay=1.01, e_max=1.00):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        self.e_decay = e_decay
        self.e_max = e_max

    def update_epsilon(self):
        # 更新 epsilon 的值，不让其降低到 e_min 以下
        if self.epsilon < self.e_max:
            self.epsilon *= self.e_decay

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            # 使用公式：Q_target = r+γ  maxQ(s',a')
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r

        # 更新公式: Q(s,a)←Q(s,a)+α(r+γ  maxQ(s',a')-Q(s,a))
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)
        self.update_epsilon()  # 更新 epsilon

# 同轨策略Sarsa
class SarsaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, e_decay=1.01, e_max=1.00):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        self.e_decay = e_decay
        self.e_max = e_max

    def update_epsilon(self):
        # 更新 epsilon 的值，不让其降低到 e_min 以下
        if self.epsilon < self.e_max:
            self.epsilon *= self.e_decay

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            # 使用公式: Q_taget = r+γQ(s',a')
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r
        # 更新公式: Q(s,a)←Q(s,a)+α(r+γQ(s',a')-Q(s,a))
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)
        self.update_epsilon()  # 更新 epsilon