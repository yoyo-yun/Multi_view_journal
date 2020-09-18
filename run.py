import cfgs.config as config
import argparse, yaml
import random
from easydict import EasyDict as edict
from common.btrainer import BTrainer
from common.trainer import Trainer
from common.mtrainer import MTrainer


def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='Bilinear Args')

    parser.add_argument('--run', dest='run_mode',
                        choices=['train', 'val', 'test', 'sample'],
                        help='{train, val, test, sample}',
                        type=str, required=True)

    parser.add_argument('--model', dest='model',
                        choices=['single', 'e2d', 'sk', 'lstm'],
                        help='{single, e2d, sk, lstm}',
                        default='e2d', type=str)

    parser.add_argument('--dataset', dest='dataset',
                        choices=['sst5', 'mr', 'trec', 'ags', 'subj', 'cr', 'reuters', 'ec'],
                        help='{sst5, mr, trec, ags}',
                        default='sst5', type=str)

    parser.add_argument('--gpu', dest='gpu',
                        help="gpu select, eg.'0, 1'",
                        type=str,
                        default="0, 1")

    parser.add_argument('--seed', dest='seed',
                        help='fix random seed',
                        type=int,
                        default=random.randint(0, 99999999))

    parser.add_argument('--version', dest='version',
                        help='version control',
                        type=str,
                        default="default")

    parser.add_argument('--trainer',
                        default='bert',
                        type=str,
                        choices=['bert', 'glove'])

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    __C = config.__C

    args = parse_args()
    cfg_file = "cfgs/{}.yml".format(args.model)
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.load(f)

    args_dict = edict({**yaml_dict, **vars(args)})
    config.add_edit(args_dict, __C)
    config.proc(__C)

    print('Hyper Parameters:')
    config.config_print(__C)

    if __C.dataset in ['ec', 'reuters']:
        execution = MTrainer(__C)
    else:
        execution = BTrainer(__C) if __C.trainer == 'bert' else Trainer(__C)
    execution.run(__C.run_mode)
