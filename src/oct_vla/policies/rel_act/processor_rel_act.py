from lerobot.policies.act.processor_act import make_act_pre_post_processors


def make_rel_act_pre_post_processors(config, dataset_stats=None):
    # Images are normalised here as for ACT; state and action pass through
    # (IDENTITY) and are normalised inside the policy.
    config.validate_features()
    return make_act_pre_post_processors(config, dataset_stats)
