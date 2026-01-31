from dataclasses import asdict
from benchmarks.configs.names import MyExperiments
from benchmarks.configs.tutorial2_experiments import surf_agent_2obj_train
from benchmarks.configs.tutorial3_experiments import surf_agent_2obj_eval
from benchmarks.configs.tutorial4_experiments import surf_agent_2obj_unsupervised
from benchmarks.configs.tutorial5_experiments import dist_agent_5lm_2obj_train
from benchmarks.configs.tutorial52_experiments import dist_agent_5lm_2obj_eval


experiments = MyExperiments(
    dist_agent_5lm_2obj_train=dist_agent_5lm_2obj_train, 
    dist_agent_5lm_2obj_eval=dist_agent_5lm_2obj_eval,
)
CONFIGS = asdict(experiments)