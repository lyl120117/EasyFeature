import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import program
from core.data import build_dataloader
from core.modeling.architectures import build_model
from core.losses import build_loss
from core.postprocess import build_post_process
from core.optimizer import build_optimizer, build_scheduler
from core.metric import build_metric
from core.utils.save_load import load_model


def main(config: dict, device, logger, vdl_writer):
    global_config = config['Global']

    # build dataloader
    train_dataloader = build_dataloader(config, 'Train', logger)
    if len(train_dataloader) == 0:
        logger.error(
            "No Images in train dataset, please ensure\n" +
            "\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n"
            +
            "\t2. The annotation file and path in the configuration file are provided normally."
        )
        return

    if config['Eval']:
        valid_dataloader = build_dataloader(config, 'Eval', logger)
    else:
        valid_dataloader = None

    if config['Test']:
        test_dataloader = build_dataloader(config, 'Test', logger)
    else:
        test_dataloader = None

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # build model
    model = build_model(config['Architecture'])

    # build loss
    loss_class = build_loss(config['Loss'])

    # build optim
    optimizer = build_optimizer(config['Optimizer'],
                                epochs=global_config['epoch_num'],
                                step_each_epoch=len(train_dataloader),
                                parameters=model.named_parameters())
    # parameters=model.parameters())

    # build scheduler
    if 'Scheduler' in config.keys():
        scheduler = build_scheduler(config['Scheduler'], optimizer,
                                    global_config['epoch_num'])
    else:
        scheduler = None

    # build metric
    eval_class = build_metric(config['Metric'])
    # load pretrain model
    pre_best_model_dict = load_model(logger, config, model, optimizer)

    logger.info('train dataloader has {} iters'.format(len(train_dataloader)))
    if valid_dataloader is not None:
        logger.info('valid dataloader has {} iters'.format(
            len(valid_dataloader)))
    if test_dataloader is not None:
        logger.info('test dataloader has {} iters'.format(
            len(test_dataloader)))

    # start train
    program.train(config, train_dataloader, valid_dataloader, test_dataloader,
                  device, model, loss_class, optimizer, scheduler,
                  post_process_class, eval_class, pre_best_model_dict, logger,
                  vdl_writer)


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    main(config, device, logger, vdl_writer)
