from collections import OrderedDict
import os


class WandbWriter:
    def __init__(self, exp_name, cfg, output_dir, cur_step=0, step_interval=0):
        try:
            import wandb
        except ImportError as exc:
            raise ImportError('启用 --use_wandb 1 前请先安装 wandb: pip install wandb') from exc

        try:
            import swanlab
        except ImportError:
            swanlab = None

        self.wandb = wandb
        self.step = cur_step
        self.interval = step_interval
        wandb_id = wandb.util.generate_id()
        if swanlab is not None:
            swanlab.sync_wandb()
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project="tracking", name=exp_name, config=cfg, dir=output_dir, resume='allow', id=wandb_id)

    def write_log(self, stats: OrderedDict, epoch=-1):
        self.step += 1
        for loader_name, loader_stats in stats.items():
            if loader_stats is None:
                continue

            log_dict = {}
            for var_name, val in loader_stats.items():
                if hasattr(val, 'avg'):
                    log_dict.update({loader_name + '/' + var_name: val.avg})
                else:
                    log_dict.update({loader_name + '/' + var_name: val.val})

                if epoch >= 0:
                    log_dict.update({loader_name + '/epoch': epoch})

            self.wandb.log(log_dict, step=self.step*self.interval)
