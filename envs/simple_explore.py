import numpy as np
from mazelab import BaseMaze
from mazelab import BaseEnv
from mazelab import Object
from mazelab import DeepMindColor as color
from mazelab.generators import random_shape_maze
from mazelab import VonNeumannMotion

import gym
from gym.spaces import Box
from gym.spaces import Discrete

import matplotlib.pyplot as plt

from abc import ABC
from abc import abstractmethod

from collections import namedtuple
from envs.env_wrappers import SimplifySubprocVecEnv
import cv2
#x = random_shape_maze(width=50, height=50, max_shapes=50, max_size=8, allow_overlap=False, shape=None)

class Maze(BaseMaze):
    @property
    def size(self):
        return x.shape
    
    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(x == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(x == 1), axis=1))
        agent = Object('agent', 2, color.agent, False, [])
        goal = Object('goal', 3, color.goal, False, [])
        return free, obstacle, agent, goal

class MAMaze(ABC):
    def __init__(self, agent_num):
        self.agent_num = agent_num
        self.env = random_shape_maze(width=60, height=60, max_shapes=60, max_size=10, allow_overlap=False, shape=None)
        objects = self.make_objects()
        assert all([isinstance(obj, Object) for obj in objects])
        self.objects = namedtuple('Objects', map(lambda x: x.name, objects), defaults=objects)()
    
    @property
    def size(self):
        r"""Returns a pair of (height, width). """
        return self.env.shape
        
    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(self.env == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(self.env == 1), axis=1))
        h, w = self.size
        agents_pos = np.zeros((self.agent_num, 2))
        #start_idx = [int(h/2), int(w/2)]
        for i in range(self.agent_num):
            agents_pos[i] = [int((h-self.agent_num)/2)+i, int(w/2)]
        agents = Object('agents', 2, color.agent, False, agents_pos.astype(np.int))
        #import pdb; pdb.set_trace()
        #agents = []
        #for i in range(self.agent_num):
        #    agents.append(Object('agent', 2, color.agent, False, []))
        return free, obstacle, agents
    
    def _convert(self, x, name):
        for obj in self.objects:
            pos = np.asarray(obj.positions)
            x[pos[:, 0], pos[:, 1]] = getattr(obj, name, None)
        return x
    
    def to_name(self):
        x = np.empty(self.size, dtype=object)
        return self._convert(x, 'name')
    
    def to_value(self):
        x = np.empty(self.size, dtype=int)
        return self._convert(x, 'value')
    
    def to_rgb(self):
        x = np.empty((*self.size, 3), dtype=np.uint8)
        return self._convert(x, 'rgb')
    
    def to_impassable(self):
        x = np.empty(self.size, dtype=bool)
        return self._convert(x, 'impassable')
    
    def __repr__(self):
        return f'{self.__class__.__name__}{self.size}'

class MAMazeEnv(BaseEnv):
    def __init__(self, agent_num):
        super().__init__()
        
        self.agent_num = agent_num
        self.maze = MAMaze(self.agent_num)
        self.gt_global_map = self.maze.to_value()
        self.motions = VonNeumannMotion()
        
        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))

        self.global_map_size = 60
        self.local_map_size = 11

        self.global_map = np.full((self.global_map_size, self.global_map_size),-1)

        self.step_n = 5

        self.episode_length = 5


    def step(self, action):
        self.timestep += 1
        info = {}
        for i in range(self.agent_num):
            motion = self.motions[action[i]]
            for j in range(self.step_n):
                current_position = self.maze.objects.agents.positions[i]
                new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
                valid = self._is_valid(new_position)
                if valid:
                    self.maze.objects.agents.positions[i] = np.asarray(new_position)
        self.gt_global_map = self.maze.to_value()
        obs = self.observation()
        reward = self.reward()
        done = self.episode_done()
        info['timestep'] = self.timestep
        return obs, reward, done, info

    def reset(self):
        self.timestep = 0
        self.maze = MAMaze(self.agent_num)
        self.gt_global_map = self.maze.to_value()
        self.global_map = np.full((self.global_map_size, self.global_map_size),-1)
        obs = self.observation()
        self.prev_explored_area = self.global_map.shape[0]**2-(self.global_map==-1).sum()
        return obs
    
    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable
    
    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out
    
    def get_image(self):
        return self.maze.to_rgb()

    def observation(self):
        local_maps = []
        global_maps = []
        for i in range(self.agent_num):        
            local_map = np.zeros((self.local_map_size, self.local_map_size))
            #fp_explored = np.zeros((self.args.local_map_size, self.args.local_map_size))
            current_pos = self.maze.objects.agents.positions[i]
            local_map = self.gt_global_map[(current_pos[0]-self.local_map_size // 2):(current_pos[0]+self.local_map_size//2+1), (current_pos[1]-self.local_map_size // 2):(current_pos[1]+self.local_map_size//2+1)]
            self.global_map[(current_pos[0]-self.local_map_size // 2):(current_pos[0]+self.local_map_size//2+1), (current_pos[1]-self.local_map_size // 2):(current_pos[1]+self.local_map_size//2+1)] = local_map
            local_maps.append(local_map)
        for i in range(self.agent_num): 
            global_maps.append(self.global_map)
        return local_maps, global_maps

    def reward(self):
        curr_explored_area = self.global_map.shape[0]**2-(self.global_map==-1).sum()
        reward = curr_explored_area - self.prev_explored_area
        self.prev_explored_area = curr_explored_area
        return reward

    def episode_done(self):
        if self.timestep >= self.episode_length:
            return True
        if self.prev_explored_area == self.global_map.shape[0]**2:
            return True
        return False


if __name__ == "__main__":
    env_id = 'RandomShapeMaze-v0'
    gym.envs.register(id=env_id, entry_point=MAMazeEnv, max_episode_steps=200)

    env = gym.make(env_id, agent_num=2)
    obs = env.reset()
    #print(obs[20:30, 20:30])
    #img = env.render('rgb_array')
    obs, reward, done, _ = env.step([1, 0])
    import pdb; pdb.set_trace()
    plt.imshow(img)
    plt.show()

    def make_parallel_env(args):
        def get_env_fn(rank):
            def init_env():
                '''
                if args.env_name == "HideAndSeek":
                    env = HideAndSeekEnv(args)
                else:
                    print("Can not support the " + args.env_name + "environment." )
                    raise NotImplementedError
                '''
                env = gym.make(args.env_id, args.agent_num)
                env.seed(args.seed + rank * 1000)
                return env
            return init_env
        return SimplifySubprocVecEnv([get_env_fn(i) for i in range(args.n_rollout_threads)])
