from lerobot.policies.act.processor_act import make_act_pre_post_processors


def make_history_act_pre_post_processors(config, dataset_stats=None):
    # Normalisation broadcasts over the extra time axis, so ACT's own pipeline
    # applies unchanged to [B, T, ...] inputs.
    config.validate_features()
    return make_act_pre_post_processors(config, dataset_stats)
