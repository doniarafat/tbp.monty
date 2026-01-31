from dataclasses import dataclass, field, fields

NAMES = []

@dataclass
class MyExperiments:
    first_experiment: dict = field(default_factory=dict)
    surf_agent_2obj_train: dict = field(default_factory=dict)
    surf_agent_2obj_eval: dict = field(default_factory=dict)
    surf_agent_2obj_unsupervised: dict = field(default_factory=dict)
    dist_agent_5lm_2obj_train: dict= field(default_factory=dict)
    dist_agent_5lm_2obj_eval:dict= field(default_factory=dict)


NAMES.extend(field.name for field in fields(MyExperiments))

@dataclass
class MontyWorldExperiments:
    world_image_from_stream_on_scanned_model: dict = field(default_factory=dict)
    world_image_on_scanned_model: dict = field(default_factory=dict)
    dark_world_image_on_scanned_model: dict = field(default_factory=dict)
    bright_world_image_on_scanned_model: dict = field(default_factory=dict)
    hand_intrusion_world_image_on_scanned_model: dict = field(default_factory=dict)
    multi_object_world_image_on_scanned_model: dict = field(default_factory=dict)


NAMES.extend(field.name for field in fields(MontyWorldExperiments))

@dataclass
class MontyWorldHabitatExperiments:
    randrot_noise_sim_on_scan_monty_world: dict = field(default_factory=dict)

NAMES.extend(field.name for field in fields(MontyWorldHabitatExperiments))

@dataclass
class PretrainingExperiments:
    supervised_pre_training_base: dict = field(default_factory=dict)
    supervised_pre_training_5lms: dict = field(default_factory=dict)
    supervised_pre_training_5lms_all_objects: dict = field(default_factory=dict)
    only_surf_agent_training_10obj: dict = field(default_factory=dict)
    only_surf_agent_training_10simobj: dict = field(default_factory=dict)
    only_surf_agent_training_allobj: dict = field(default_factory=dict)
    only_surf_agent_training_numenta_lab_obj: dict = field(default_factory=dict)

NAMES.extend(field.name for field in fields(PretrainingExperiments))

@dataclass
class YcbExperiments:
    base_config_10distinctobj_dist_agent: dict = field(default_factory=dict)
    base_config_10distinctobj_surf_agent: dict = field(default_factory=dict)
    randrot_noise_10distinctobj_dist_agent: dict = field(default_factory=dict)
    randrot_noise_10distinctobj_dist_on_distm: dict = field(default_factory=dict)
    randrot_noise_10distinctobj_surf_agent: dict = field(default_factory=dict)
    randrot_10distinctobj_surf_agent: dict = field(default_factory=dict)
    randrot_noise_10distinctobj_5lms_dist_agent: dict = field(default_factory=dict)
    base_10simobj_surf_agent: dict = field(default_factory=dict)
    randrot_noise_10simobj_surf_agent: dict = field(default_factory=dict)
    randrot_noise_10simobj_dist_agent: dict = field(default_factory=dict)
    randomrot_rawnoise_10distinctobj_surf_agent: dict = field(default_factory=dict)
    base_10multi_distinctobj_dist_agent: dict = field(default_factory=dict)
    surf_agent_unsupervised_10distinctobj: dict = field(default_factory=dict)
    surf_agent_unsupervised_10distinctobj_noise: dict = field(default_factory=dict)
    surf_agent_unsupervised_10simobj: dict = field(default_factory=dict)
    base_77obj_dist_agent: dict = field(default_factory=dict)
    base_77obj_surf_agent: dict = field(default_factory=dict)
    randrot_noise_77obj_surf_agent: dict = field(default_factory=dict)
    randrot_noise_77obj_dist_agent: dict = field(default_factory=dict)
    randrot_noise_77obj_5lms_dist_agent: dict = field(default_factory=dict)

NAMES.extend(field.name for field in fields(YcbExperiments))

@dataclass
class UnsupervisedInferenceExperiments:
    unsupervised_inference_distinctobj_surf_agent: dict = field(default_factory=dict)
    unsupervised_inference_distinctobj_dist_agent: dict = field(default_factory=dict)

NAMES.extend(field.name for field in fields(UnsupervisedInferenceExperiments))
