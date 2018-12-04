#!/usr/bin/env python3

import numpy as np
import gym


class PidController:
    cum_err = 0
    prev_err = 0
    kp = 0
    ki = 0
    kd = 0

    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def step(self, err) -> float:
        self.cum_err += (err + self.prev_err) / 2.0

        p = self.kp * err
        i = self.ki * self.cum_err
        d = self.kd * 0

        print("%+00.5f %+00.5f %+00.5f  %+00.5f  %+00.5f" % (err, p, self.cum_err, i, p + i + d))

        self.prev_err = err

        return p + i + d


def run():
    env = gym.make('CartPole-v0').env
    env.theta_threshold_radians = 10
    for i_episode in range(2):
        observation = env.reset()

        goal_theta = 0
        goal_x = 0
        goal_accel = 0
        pid = PidController(1, 0.08, 0)

        for t in range(1000):
            env.render()
            _, accel, theta, x = observation
            err = goal_theta - theta
            desired_output = pid.step(err)
            action = 0 if desired_output > 0 else 1
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


if __name__ == '__main__':
    run()
