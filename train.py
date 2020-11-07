from envs.simple_explore import MAMazeEnv
from config import get_config
from envs.env_wrappers import SimplifySubprocVecEnv, DummyVecEnv
import gym

def make_parallel_env(args):
    def get_env_fn(rank):
        def init_env():
            env = gym.make(args.env_name, agent_num=args.num_agents)
            env.seed(args.seed + rank * 1000)
            return env
        return init_env
    return SimplifySubprocVecEnv([get_env_fn(i) for i in range(args.n_rollout_threads)])

def main():
    args = get_config()
    gym.envs.register(id=args.env_name, entry_point=MAMazeEnv, max_episode_steps=200)
    envs = make_parallel_env(args)
    obs = envs.reset()
    obs,_,_,_ = envs.step([[0,1] for i in range(args.n_rollout_threads)])
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
