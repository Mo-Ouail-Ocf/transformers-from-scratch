from ignite.engine import Engine,Events
from ignite.handlers.tqdm_logger import ProgressBar
from ignite.handlers.tensorboard_logger import TensorboardLogger,OutputHandler
from ignite.metrics import Loss , RunningAverage
from ignite.handlers.checkpoint import Checkpoint,DiskSaver
from ignite.metrics.nlp import Bleu
import torch.nn as nn
import torch

from config import Config

def attach_ignite(
        trainer:Engine,
        evaluator:Engine,
        model:nn.Module,
        optimizer:torch.optim.Adam,
        loss_fn:nn.Module,
        valid_dl
):
    # trainer : attach tqdm + running avergae

    config =Config()
    
    running_avg_loss = RunningAverage(
        output_transform=lambda x : x['loss']
    ).attach(trainer,name='avg_loss')
    tqdm_logger_trainer = ProgressBar().attach(
        engine=trainer,
        output_transform= lambda out : out['loss'],
    )

    # Evaluator , attach loss metric
    loss_metric = Loss(loss_fn)
    loss_metric.attach(
        engine=evaluator,
        name='loss',
    )

    tqdm_logger_eval = ProgressBar().attach(
        engine=evaluator,
        metric_names='all',
    )

    # logging for tb logger

    tb_logger = TensorboardLogger(log_dir="logs")
    
    trainer_out_handler = OutputHandler(tag='train',
                                        output_transform=lambda x:x['loss'],    
                                        )
    tb_logger.attach(
        engine=trainer,
        log_handler=trainer_out_handler,
        event_name=Events.EPOCH_COMPLETED
    )

    valid_log_handler = OutputHandler(tag='valid',
                                        metric_names='all',    
                                        )
    tb_logger.attach(
        engine=evaluator,
        log_handler=valid_log_handler,
        event_name=Events.EPOCH_COMPLETED,
    )

    # valid :

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_valid(engine):
        print(f'~~~ Epoch {trainer.state.epoch} completed ~~~~~~~~~~')
        evaluator.run(valid_dl)
        print(f'-> Average train loss : {trainer.state.metrics['avg_loss']}')
        print(f'-> Average validation loss : {evaluator.state.metrics['loss']}')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    # Checkpointing
    to_save = {
        'model':model,
        'trainer':trainer,
        'optimizer':optimizer
    }

    checkpoint_handler = Checkpoint(
        to_save=to_save,
        save_handler=DiskSaver(
            dirname=config.model_folder,
            require_empty=False
        ),
        n_saved=2,
        filename_prefix='checkpoint'
    )

    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED,
        handler=checkpoint_handler
    )
    
def resume_from_checkpoint(trainer, model, optimizer, config):
    # Load latest checkpoint
    checkpoint_fp = config.latest_weights_file_path()
    if checkpoint_fp is not None:
        checkpoint = torch.load(checkpoint_fp)
        Checkpoint.load_objects(
            to_load={'model': model, 'optimizer': optimizer, 'trainer': trainer},
            checkpoint=checkpoint
        )