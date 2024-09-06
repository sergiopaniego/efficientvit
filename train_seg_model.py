import argparse
import os

import wandb

from efficientvit.apps import setup
from efficientvit.apps.utils import dump_config, parse_unknown_args
from efficientvit.seg_model_zoo import create_seg_model
from efficientvit.segcore.data_provider import SEGDataProvider 
from efficientvit.segcore.trainer import SEGRunConfig, SEGTrainer ###

parser = argparse.ArgumentParser()
parser.add_argument("config", metavar="FILE", help="config file")
parser.add_argument("--path", type=str, metavar="DIR", help="run directory")
parser.add_argument("--manual_seed", type=int, default=0)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--amp", type=str, choices=["fp32", "fp16", "bf16"], default="fp32")
parser.add_argument("--rand_init", type=str, default="trunc_normal@0.02")
parser.add_argument("--last_gamma", type=float, default=0)


def main():
    args, opt = parser.parse_known_args()
    opt = parse_unknown_args(opt)

    #setup.setup_dist_env()p

    os.makedirs(args.path, exist_ok=True)
    dump_config(args.__dict__, os.path.join(args.path, "args.yaml"))

    #setup.setup_seed(args.manual_seed, args.resume)

    config = setup.setup_exp_config(args.config, recursive=True, opt_args=opt)

    #setup.save_exp_config(config, args.path)

    print(config)

    #data_provider = setup.setup_data_provider(config, [SEGDataProvider], is_distributed=True) ###
    data_provider = setup.setup_data_provider(config, [SEGDataProvider], is_distributed=False) ###


    run_config = setup.setup_run_config(config, SEGRunConfig) ###

    #model = create_seg_model(config["net_config"]["name"], False) ###
    model = create_seg_model(config["net_config"]["name"], 'cityscapes')

    trainer = SEGTrainer( ###
        path=args.path,
        model=model,
        data_provider=data_provider,
        project_name=config['net_config']['project_name']
    )

    setup.init_model(
        trainer.network,
        init_from=config["net_config"]["ckpt"],
        rand_init=args.rand_init,
        last_gamma=args.last_gamma,
    )

    trainer.prep_for_training(run_config, args.amp)

    if args.resume:
        trainer.load_model()
        trainer.data_provider = setup.setup_data_provider(config, [SEGDataProvider], is_distributed=True) ###
    else:
        pass
        #trainer.sync_model()

    trainer.train()


if __name__ == "__main__":
    main()
