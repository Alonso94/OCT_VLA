from lerobot.policies.act.processor_act import make_act_pre_post_processors


def make_aug_act_pre_post_processors(config, dataset_stats=None):
    config.validate_features()
    return make_act_pre_post_processors(config, dataset_stats)
