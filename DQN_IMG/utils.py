import cv2
import gym
import numpy
import time


class FrameStackingAndResizingEnv:
    def __init__(self, env: gym.Env, w, h, num_stack=4):
        self.env = env
        self.w = w
        self.h = h
        self.num_stack = num_stack
        self.buffer = numpy.zeros(shape=(self.num_stack, self.h, self.w),
                                  dtype='uint8')

    def _preprocess_fram(self, frame):
        image = cv2.resize(frame, (self.w, self.h))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def render(self):
        self.env.render()

    def reset(self):
        im = self.env.reset()
        im = self._preprocess_fram(im)
        self.buffer = numpy.stack([im] * self.num_stack, 0)
        return self.buffer.copy()

    def step(self, action):
        im, reward, done, info = self.env.step(action)
        im = self._preprocess_fram(im)
        self.buffer[1:self.num_stack, :, :] = self.buffer[0:self.num_stack -
                                                          1, :, :]
        self.buffer[0, :, :] = im
        return self.buffer.copy(), reward, done, info

    @property
    def observation_space(self):
        return numpy.zeros((self.num_stack, self.h, self.w))

    @property
    def action_space(self):
        return self.env.action_space


if __name__ == "__main__":
    env = gym.make('Breakout-v0')
    env = FrameStackingAndResizingEnv(env, 480, 640)
    env.reset()
    env.step(1)
    for j in range(30):
        action = env.env.action_space.sample()
        img, reward, done, _ = env.step(action)
        ims = []
        for i in range(img.shape[-1]):
            ims.append(img[:, :, i])
        cv2.imwrite(f"/tmp/{j}.jpg", numpy.hstack(ims))
        time.sleep(0.1)
        if done:
            break
