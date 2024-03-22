from musenet import MuseNetModel
import argparse
from utils import instantiate_from_config
from omegaconf import OmegaConf
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  '-c',
                        help='path to the config file',
                        default='configs/musenet_train.yaml')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # parse args and load config
    args = arg_parse()
    config = OmegaConf.load(args.config)
    # init datamodule
    datamodule = instantiate_from_config(config.datamodule)
    datamodule.setup()
    # init model
    model = instantiate_from_config(config.model)
    # log
    logger = TensorBoardLogger(**config.logging)
    seed_everything(config.seed, True)
    # checkpoint
    checkpoint_callback = ModelCheckpoint(save_top_k=config.ckpt_save_top_k, monitor="val_loss")
    # trainer
    trainer = Trainer(logger=logger, **config.trainer, callbacks=[checkpoint_callback])
    # train
    trainer.fit(model, datamodule=datamodule)
    
    