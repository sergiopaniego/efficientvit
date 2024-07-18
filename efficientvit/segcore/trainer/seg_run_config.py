from efficientvit.apps.trainer.run_config import RunConfig

__all__ = ["SEGRunConfig"]


class SEGRunConfig(RunConfig):
    @property
    def none_allowed(self):
        return ["reset_bn", "reset_bn_size", "reset_bn_batch_size"] + super().none_allowed
