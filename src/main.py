import utils
import flappy_bird_gym
import random
import time


import numpy as np

# alpha: learning rate (0.1 - 0.0001)
# epsilon: exploration-exploitation trade-off (0.1 - 0.5)
# landa(lambda): discount factor (0.9 - 0.99)

class SmartFlappyBird:
    def __init__(self, iterations):
        self.Qvalues = utils.Counter()
        self.landa = 0.99
        self.epsilon = 0.5  # change to proper value
        self.alpha = 0.03  # change to proper value
        self.iterations = iterations

    def policy(self, state):
        # implement the policy

        actions = self.get_all_actions()

        if (utils.flip_coin(self.epsilon)):
            return random.choice(actions)
        else:
            return self.max_arg(state)

        # return NotImplemented
 
    @staticmethod
    def get_all_actions():
        return [0, 1]

    @staticmethod
    def convert_continuous_to_discrete(state):
        # implement the best way to convert continuous distance values to discrete values

        print(state, end=" ")
        
        horiz_dist_pipe = np.digitize(state[0], bins = np.linspace(0, 1.7, num = 8))
        y_dist_hole = np.digitize(state[1], bins = np.linspace(-0.8, 0.8, num = 10))
        print((horiz_dist_pipe, y_dist_hole))
        
        return (horiz_dist_pipe, y_dist_hole)

        # return NotImplemented

    def compute_reward(self, prev_info, new_info, done, observation):
        # implement the best way to compute reward based on observation and score

        reward = 0

        if (new_info["score"] > prev_info["score"]):
            reward += 1
        
        if (done and (observation[0] == 0 and observation[1] != 0)):
            reward -= 10
        elif (not done):
            reward = self.landa * self.maxQ(observation)
        
        return reward

        # return NotImplemented

    def get_action(self, state):
        # implement the best way to get action based on current state
        # you can use `utils.flip_coin` and `random.choices`
        
        discreteState = self.convert_continuous_to_discrete(state)
        selectedAction = self.policy(discreteState)
        return selectedAction

        # return NotImplemented

    def maxQ(self, state):
        # return max Q value of a state

        actions = self.get_all_actions()
        MaxQValue = float(-np.inf)

        for action in actions:
            QValue = self.Qvalues[(tuple(state), action)]
            if (QValue > MaxQValue):
                MaxQValue = QValue

        return MaxQValue
    
        # return NotImplemented

    def max_arg(self, state):
        # return argument of the max q of a state

        actions = self.get_all_actions()

        max_action = actions[0]
        max_q = self.Qvalues.get((state, max_action), 0)

        for action in actions:
            q_value = self.Qvalues.get((state, action), 0)
            if (q_value > max_q):
                max_action = action
                max_q = q_value

        return max_action
    
        # return NotImplemented

    def update(self, reward, state, action, next_state):
        # update q table

        CurrentQValue = self.Qvalues.get((tuple(state), action), 0)
        MaxFutureQValue = self.maxQ(next_state)

        UpdatedQValue = ((1 - self.alpha) * CurrentQValue) + (self.alpha * (reward + self.landa * MaxFutureQValue))

        self.Qvalues[(tuple(state), action)] = UpdatedQValue

        # return NotImplemented

    def update_epsilon_alpha(self):
        # update epsilon and alpha base on iterations

        self.epsilon = max(0.1, self.epsilon * 0.96)
        self.alpha = max(0.00001, self.alpha * 0.955)

        # return NotImplemented

    def run_with_policy(self, landa):
        self.landa = landa
        env = flappy_bird_gym.make("FlappyBird-v0")
        observation = env.reset() # horizontal dist from the next pipe, distance from the next hole
        info = {'score': 0}
        for _ in range(self.iterations):
            while True:
                action = self.get_action(observation)  # policy affects here
                this_state = observation
                prev_info = info
                observation, reward, done, info = env.step(action)
                reward = self.compute_reward(prev_info, info, done, observation)
                self.update(reward, this_state, action, observation)
                self.update_epsilon_alpha()
                if done:
                    observation = env.reset()
                    break
        env.close()

    def run_with_no_policy(self, landa):
        self.landa = landa
        # no policy test
        env = flappy_bird_gym.make("FlappyBird-v0")
        observation = env.reset()
        info = {'score': 0}
        while True:
            action = self.get_action(observation)
            prev_info = info
            observation, reward, done, info = env.step(action)
            reward = self.compute_reward(prev_info, info, done, observation)
            env.render()
            time.sleep(1 / 30)  # FPS
            if done:
                break
        env.close()

    def run(self):
        self.run_with_policy(1)
        self.run_with_no_policy(1)

program = SmartFlappyBird(iterations=100)
program.run()


