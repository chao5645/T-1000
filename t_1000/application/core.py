import ray
from t_1000.application.handlers import find_results_folder, get_instruments_from_checkpoint
from utils.data_processing import get_datasets
from ray import tune
from t_1000.env.trading_env import TradingEnv
from ray.tune.registry import register_env
from ray.rllib.agents.registry import get_agent_class
import collections
import gym
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.evaluation.episode import flatten_to_single_ndarray

env_name = 'YesMan-v1'


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(_):
    return DEFAULT_POLICY_ID


def rollout(agent, env_name, num_steps, no_render=True):
    policy_agent_mapping = default_policy_agent_mapping

    print("agent: {} env_name: {}".format(agent, env_name))
    if hasattr(agent, "workers"):
        env = agent.workers.local_worker().env_creator(agent.workers.local_worker().env_context)
        print("local_work: {}".format(agent.workers.local_worker()))
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state()
                      for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        action_init = {
            p: flatten_to_single_ndarray(m.action_space.sample())
            for p, m in policy_map.items()
        }
    else:
        env = gym.make(env_name)
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    steps = 0
    while steps < (num_steps or steps + 1):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic

        print(env)
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        while not done and steps < (num_steps or steps + 1):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                    a_action = flatten_to_single_ndarray(a_action)  # tuple actions
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, _ = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward
            if not no_render:
                env.render()
            steps += 1
            obs = next_obs


class T1000:
    def __init__(self, algo, assets, currency, granularity, datapoints, checkpoint_path, initial_account_balance,
                 exchange_commission, exchange):
        self.algo = algo
        self.assets = assets
        self.currency = currency
        self.granularity = granularity
        self.datapoints = datapoints
        self.df = {}
        self.config_spec = {}
        self.initial_account_balance = initial_account_balance
        self.exchange_commission = exchange_commission
        if checkpoint_path:
            _, self.assets, self.currency, self.datapoints, self.granularity = get_instruments_from_checkpoint(
                checkpoint_path)
        self.check_variables_integrity()
        self.populate_dfs(exchange=exchange)
        self.config_spec_variables = {
            "candlestick_width": {  # constants
                "day": 1,
                "hour": 0.04,
                "minute": 0.0006
            },
            "initial_account_balance": self.initial_account_balance,
            "commission": self.exchange_commission
        }

    def trial_name_string(self, trial):
        return '{}_{}_{}_{}'.format('-'.join(self.assets), self.currency, self.granularity, self.datapoints)

    def check_variables_integrity(self):
        if type(self.assets) != list or len(self.assets) == 0:
            raise ValueError("Incorrect 'assets' value")
        if type(self.currency) != str:
            raise ValueError("Incorrect 'currency' value")
        if type(self.granularity) != str:
            raise ValueError("Incorrect 'granularity' value")
        if type(self.datapoints) != int or 1 > self.datapoints > 2000:
            raise ValueError("Incorrect 'datapoints' value")

    def populate_dfs(self, exchange):
        for asset in self.assets:
            self.df[asset] = {}
            self.df[asset]['train'], self.df[asset]['rollout'] = get_datasets(asset=asset,
                                                                              currency=self.currency,
                                                                              granularity=self.granularity,
                                                                              datapoints=self.datapoints,
                                                                              exchange=exchange)

    def generate_config_spec(self, lr_schedule, df_type):
        self.config_spec = {
            "lr_schedule": tune.grid_search(lr_schedule),
            "env": env_name,
            "num_workers": 6,  # parallelism
            "num_gpus": 0,
            'observation_filter': 'MeanStdFilter',
            'vf_share_layers': True,
            "env_config": {
                'assets': self.assets,
                'currency': self.currency,
                'granularity': self.granularity,
                'datapoints': self.datapoints,
                'df_complete': {},
                'df_features': {},
                'variables': self.config_spec_variables
            },
        }
        self.add_dfs_to_config_spec(df_type=df_type)

    def add_dfs_to_config_spec(self, df_type):
        for asset in self.assets:
            self.config_spec['env_config']['df_complete'][asset] = self.df[asset][df_type]
            self.config_spec['env_config']['df_features'][asset] = self.df[asset][df_type].loc[:,
                                                                   self.df[asset][df_type].columns != 'Date']

    def backtest(self, checkpoint_path):
        agent_config, assets, currency, datapoints, granularity = get_instruments_from_checkpoint(
            checkpoint_path)

        config = {
            'assets': assets,
            'currency': currency,
            'granularity': granularity,
            'datapoints': datapoints,
            'df_complete': {},
            'df_features': {},
            'variables': self.config_spec_variables
        }

        for asset in assets:
            config['df_complete'][asset] = self.df[asset]['rollout']
            config['df_features'][asset] = self.df[asset]['rollout'].loc[:,
                                           self.df[asset]['rollout'].columns != 'Date']

        register_env(env_name, lambda _: TradingEnv(config))
        ray.init()
        cls = get_agent_class(self.algo)
        agent = cls(env=env_name, config=agent_config)
        agent.restore(checkpoint_path)

        num_steps = int(len(config['df_complete'][assets[0]]))
        no_render = False

        rollout(agent, env_name, num_steps, no_render)

    def train(self, timesteps, checkpoint_freq, lr_schedule):
        register_env(env_name, lambda config: TradingEnv(config))
        #ray.init(address="192.168.100.149:6379", _redis_password="5241590000000000")
        ray.init()

        self.generate_config_spec(lr_schedule=lr_schedule, df_type='train')

        tune.run(name=self.algo,
                 run_or_experiment=self.algo,
                 checkpoint_at_end=True,
                 stop={'timesteps_total': timesteps},
                 checkpoint_freq=checkpoint_freq,
                 config=self.config_spec,
                 local_dir=find_results_folder(),
                 trial_name_creator=self.trial_name_string)
