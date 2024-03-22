import torch
import lightning.pytorch as pl
from musenet import MuseNetPipeline
import argparse
from utils import instantiate_from_config
from omegaconf import OmegaConf

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  '-c',
                        help='path to the config file',
                        default='configs/musenet_inference.yaml')
    
    parser.add_argument('--audio', '-a',
                        help='path to analysis audio', required=True)
    
    parser.add_argument('--gpu', '-g',
                        help='use gpu or not', action='store_true')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # parse args and load config
    args = arg_parse()
    config = OmegaConf.load(args.config)
    # change config using argparse
    if args.gpu:
        config.pipeline.params.device = 'cuda'
    else:
        config.pipeline.params.device = 'cpu'
    # init MuseNetPipeline 
    pipeline = instantiate_from_config(config.pipeline)
    # analysis audio
    print("Analysing Audio:", args.audio)
    pipeline.pipe(args.audio)
    

