import os
import time
import platform
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import yaml
import torch
from tqdm import tqdm
import numpy as np
import random

from core.utils.logging import get_logger
from core.utils.utility import print_dict
from core.utils.stats import TrainingStats
from core.utils.save_load import save_model
from core.data import build_dataloader


class ArgsParser(ArgumentParser):

    def __init__(self):
        super(ArgsParser,
              self).__init__(formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c", "--config", help="configuration file to use")
        self.add_argument("-o",
                          "--opt",
                          nargs='+',
                          help="set configuration options")
        self.add_argument(
            '-p',
            '--profiler_options',
            type=str,
            default=None,
            help=
            'The option of profiler, which should be in format \"key1=value1;key2=value2;key3=value3\".'
        )

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=')
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config


class AttrDict(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, **kwargs):
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))


global_config = AttrDict()

default_config = {
    'Global': {
        'debug': False,
    }
}


def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    merge_config(default_config)
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    merge_config(yaml.load(open(file_path, 'rb'), Loader=yaml.Loader))
    return global_config


def merge_config(config):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in config.items():
        if "." not in key:
            if isinstance(value, dict) and key in global_config:
                global_config[key].update(value)
            else:
                global_config[key] = value
        else:
            sub_keys = key.split('.')
            assert (
                sub_keys[0] in global_config
            ), "the sub_keys can only be one of global_config: {}, but get: {}, please check your running command".format(
                global_config.keys(), sub_keys[0])
            cur = global_config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]


def init_seed(logger, seed):
    """
    Initialize the seed for random number generator.
    Args:
        seed (int): The seed to be set.
    """
    logger.info('Set random seed to {}'.format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def preprocess(is_train=False):
    FLAGS = ArgsParser().parse_args()
    profiler_options = FLAGS.profiler_options
    config = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    profile_dic = {"profiler_options": FLAGS.profiler_options}
    merge_config(profile_dic)

    device = "cuda" if config['Global']['use_gpu'] and torch.cuda.is_available(
    ) else "cpu"

    seed = config['Global'].get('seed', None)
    if seed is None:
        seed = random.randint(10000, 100000)
        config['Global']['seed'] = seed

    if is_train:
        # save_config
        save_model_dir = config['Global']['save_model_dir']
        os.makedirs(save_model_dir, exist_ok=True)
        with open(os.path.join(save_model_dir, 'config.yml'), 'w') as f:
            yaml.dump(dict(config),
                      f,
                      default_flow_style=False,
                      sort_keys=False)
        log_file = '{}/train.log'.format(save_model_dir)
    else:
        log_file = None
    logger = get_logger(name='root', log_file=log_file)

    print_dict(config, logger)
    logger.info('train with torch {} and device {}'.format(
        torch.__version__, device))

    # set seed
    init_seed(logger, seed)

    if config['Global']['use_visualdl']:
        from visualdl import LogWriter
        save_model_dir = config['Global']['save_model_dir']
        vdl_writer_path = '{}/vdl/'.format(save_model_dir)
        os.makedirs(vdl_writer_path, exist_ok=True)
        vdl_writer = LogWriter(logdir=vdl_writer_path)
    else:
        vdl_writer = None

    return config, device, logger, vdl_writer


def train(config,
          train_dataloader,
          valid_dataloader,
          test_dataloader,
          device,
          model,
          loss_class,
          optimizer,
          scheduler,
          post_process_class,
          eval_class,
          pre_best_model_dict,
          logger,
          vdl_writer=None):
    global_config = config['Global']
    cal_metric_during_train = global_config.get('cal_metric_during_train',
                                                False)
    log_smooth_window = global_config['log_smooth_window']
    epoch_num = global_config['epoch_num']
    print_batch_step = global_config['print_batch_step']
    eval_batch_step = global_config['eval_batch_step']
    save_epoch_step = global_config['save_epoch_step']
    save_model_dir = global_config['save_model_dir']

    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    main_indicator = eval_class.main_indicator
    best_model_dict = {main_indicator: 0}
    best_test_dict = {
        'best_epoch': 0,
        main_indicator: 0,
    }

    train_stats = TrainingStats(log_smooth_window, ['lr'])
    if 'start_epoch' in pre_best_model_dict:
        start_epoch = pre_best_model_dict['start_epoch']
    else:
        start_epoch = 1
    global_step = 0
    if 'global_step' in pre_best_model_dict:
        global_step = pre_best_model_dict['global_step']

    start_eval_step = 0
    if type(eval_batch_step) == list and len(eval_batch_step) >= 2:
        start_eval_step = eval_batch_step[0]
        eval_batch_step = eval_batch_step[1]
        if len(valid_dataloader) == 0:
            logger.info(
                'No Images in eval dataset, evaluation during training will be disabled'
            )
            start_eval_step = 1e111
        logger.info(
            "During the training process, after the {}th iteration, an evaluation is run every {} iterations"
            .format(start_eval_step, eval_batch_step))
    model.train()
    model = model.to(device)

    ps = getattr(model.head, 'ps', None)
    if ps is not None:
        logger.info('ps: {:.5f}'.format(ps.data.cpu().detach().numpy()[0]))

    for epoch in range(start_epoch, epoch_num + 1):
        train_dataloader = build_dataloader(config, 'Train', logger)
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()
        max_iter = len(train_dataloader) - 1 if platform.system(
        ) == "Windows" else len(train_dataloader)
        for idx, batch in enumerate(train_dataloader):
            train_reader_cost += time.time() - reader_start
            if idx >= max_iter:
                break
            if scheduler is not None:
                lr = scheduler.get_last_lr()[0]
            else:
                lr = optimizer.defaults['lr']
            batch[0] = batch[0].to(device)
            batch[1] = batch[1].to(device)
            images = batch[0]

            train_start = time.time()
            preds = model(images, batch[1])
            loss = loss_class(preds, batch)
            avg_loss = loss['loss']

            # Backpropagation
            optimizer.zero_grad()
            avg_loss.backward()
            optimizer.step()

            train_run_cost += time.time() - train_start
            total_samples += len(images)

            # logger and visualdl
            stats = {
                k: v.cpu().detach().numpy().mean()
                for k, v in loss.items()
            }
            stats['lr'] = lr
            train_stats.update(stats)

            if cal_metric_during_train:
                batch = [item.cpu().numpy() for item in batch]
                post_result = post_process_class(preds, batch[1])
                eval_class(post_result, batch)
                metric = eval_class.get_metric()
                train_stats.update(metric)

            # vdl_writer
            if vdl_writer is not None:
                for k, v in train_stats.get().items():
                    vdl_writer.add_scalar('TRAIN/{}'.format(k), v, global_step)
                vdl_writer.add_scalar('TRAIN/lr', lr, global_step)

            if (global_step > 0 and global_step % print_batch_step
                    == 0) or (idx >= len(train_dataloader) - 1):
                logs = train_stats.log()
                if ps is not None:
                    ps_str = ' '.join(['{:.5f}'.format(p) for p in ps])
                    logs += ', ps: {}'.format(ps_str)
                strs = 'epoch: [{}/{}], iter: {}, {}, reader_cost: {:.5f} s, batch_cost: {:.5f} s, samples: {}, ips: {:.5f}'.format(
                    epoch, epoch_num, global_step, logs,
                    train_reader_cost / print_batch_step,
                    (train_reader_cost + train_run_cost) / print_batch_step,
                    total_samples,
                    total_samples / (train_reader_cost + train_run_cost))
                logger.info(strs)
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0

            # eval
            if global_step > start_eval_step and \
                    (global_step - start_eval_step) % eval_batch_step == 0:
                cur_metric = eval(model, valid_dataloader, post_process_class,
                                  eval_class, device)
                cur_metric_str = 'cur metric, {}'.format(', '.join(
                    ['{}: {}'.format(k, v) for k, v in cur_metric.items()]))
                logger.info(cur_metric_str)

                # logger metric
                if vdl_writer is not None:
                    for k, v in cur_metric.items():
                        if isinstance(v, (float, int)):
                            vdl_writer.add_scalar('EVAL/{}'.format(k),
                                                  cur_metric[k], global_step)
                test_metric_str = None
                if cur_metric[main_indicator] >= best_model_dict[
                        main_indicator]:
                    best_model_dict.update(cur_metric)
                    best_model_dict['best_epoch'] = epoch
                    save_model(model,
                               optimizer,
                               save_model_dir,
                               logger,
                               is_best=True,
                               prefix='best_accuracy',
                               best_model_dict=best_model_dict,
                               epoch=epoch,
                               global_step=global_step)

                    test_metric = eval(model, test_dataloader,
                                       post_process_class, eval_class, device)
                    if test_metric[main_indicator] >= best_test_dict[
                            main_indicator]:
                        best_test_dict.update(test_metric)
                        best_test_dict['best_epoch'] = epoch
                        save_model(model,
                                   optimizer,
                                   save_model_dir,
                                   logger,
                                   is_best=True,
                                   prefix='best_test_accuracy',
                                   best_model_dict=best_test_dict,
                                   epoch=epoch,
                                   global_step=global_step)
                    test_metric_str = 'test metric, {}'.format(', '.join([
                        '{}: {}'.format(k, v) for k, v in test_metric.items()
                    ]))
                best_str = 'best metric, {}'.format(', '.join([
                    '{}: {}'.format(k, v) for k, v in best_model_dict.items()
                ]))
                logger.info(best_str)
                if test_metric_str is not None:
                    logger.info(test_metric_str)

                best_test_metric_str = 'best test metric, {}'.format(', '.join(
                    ['{}: {}'.format(k, v)
                     for k, v in best_test_dict.items()]))
                logger.info(best_test_metric_str)
                # logger best metric
                if vdl_writer is not None:
                    vdl_writer.add_scalar(
                        'EVAL/best_{}'.format(main_indicator),
                        best_model_dict[main_indicator], global_step)
            global_step += 1
            reader_start = time.time()

        if scheduler is not None:
            scheduler.step()
        save_model(model,
                   optimizer,
                   save_model_dir,
                   logger,
                   is_best=False,
                   prefix='latest',
                   best_model_dict=best_model_dict,
                   epoch=epoch,
                   global_step=global_step)
        if epoch > 0 and epoch % save_epoch_step == 0:
            path = os.path.join(save_model_dir, 'checkpoints')
            if not os.path.exists(path):
                os.makedirs(path)
            save_model(model,
                       optimizer,
                       path,
                       logger,
                       is_best=False,
                       prefix='iter_epoch_{}'.format(epoch),
                       best_model_dict=best_model_dict,
                       epoch=epoch,
                       global_step=global_step)
    best_str = 'best metric, {}'.format(', '.join(
        ['{}: {}'.format(k, v) for k, v in best_model_dict.items()]))
    logger.info(best_str)
    if vdl_writer is not None:
        vdl_writer.close()
    return


def eval(model, valid_dataloader, post_process_class, eval_class, device):
    model.eval()
    count = 0
    with torch.no_grad():
        total_frame = 0.0
        total_time = 0.0
        pbar = tqdm(total=len(valid_dataloader),
                    desc='eval model:',
                    position=0,
                    leave=True)
        max_iter = len(valid_dataloader) - 1 if platform.system(
        ) == "Windows" else len(valid_dataloader)
        for idx, batch in enumerate(valid_dataloader):
            if idx > max_iter:
                break
            batch[0] = batch[0].to(device)
            batch[1] = batch[1].to(device)
            images = batch[0]
            count += len(images)
            start = time.time()

            preds = model(images)
            batch = [item.cpu().numpy() for item in batch]

            # Obtain usable results from post-processing methods
            total_time += time.time() - start

            # Evaluate the results of the current batch
            post_result = post_process_class(preds, batch[1])
            eval_class(post_result, batch)

            pbar.update(1)
            total_frame += len(images)
        # Get final metric，eg. acc or hmean
        metric = eval_class.get_metric()

    pbar.close()
    model.train()
    metric['count'] = count
    metric['fps'] = total_frame / total_time
    return metric