import os
import errno
import torch

__all__ = ['load_model', 'save_model']


def _mkdir_if_not_exist(path, logger):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    'be happy if some process has already created {}'.format(
                        path))
            else:
                raise OSError('Failed to mkdir {}'.format(path))


def load_model(logger, config, model, optimizer=None):
    global_config = config['Global']
    checkpoints = global_config.get('checkpoints')
    pretrained_model = global_config.get('pretrained_model')
    best_model_dict = {}
    if checkpoints:
        if not checkpoints.endswith('.pth'):
            assert os.path.exists(
                checkpoints), "The {}.pth does not exists!".format(checkpoints)
        params = torch.load(checkpoints)
        optim_dict = params['optimizer']
        optimizer.load_state_dict(optim_dict)

        pre_state_dict = params['state_dict']
        state_dict = model.state_dict()
        new_state_dict = {}
        for key, value in state_dict.items():
            if key not in pre_state_dict:
                logger.warning(
                    "{} not in loaded params, just ignore!".format(key))
                continue
            pre_value = pre_state_dict[key]
            if list(value.shape) == list(pre_value.shape):
                new_state_dict[key] = pre_value
            else:
                logger.warning(
                    "The shape of model params {} {} not matched with loaded params shape {} !"
                    .format(key, value.shape, pre_value.shape))
        if 'epoch' in params:
            best_model_dict['start_epoch'] = params['epoch'] + 1
        if 'global_step' in params:
            best_model_dict['global_step'] = params['global_step']
        logger.info("resume from {}".format(checkpoints))
        model.load_state_dict(new_state_dict)
    elif pretrained_model:
        load_pretrained_params(logger, model, pretrained_model)
    else:
        logger.info('train from scratch')
    return best_model_dict


def load_pretrained_params(logger, model, path):
    if not path.endswith('.pth'):
        assert os.path.exists(path), "The {}.pth does not exists!".format(path)
    params = torch.load(path)
    pre_state_dict = params['state_dict']
    state_dict = model.state_dict()
    new_state_dict = {}
    for key, value in state_dict.items():
        if key not in pre_state_dict:
            logger.warning("{} not in loaded params, just ignore!".format(key))
            continue
        pre_value = pre_state_dict[key]
        if list(value.shape) == list(pre_value.shape):
            new_state_dict[key] = pre_value
        else:
            logger.warning(
                "The shape of model params {} {} not matched with loaded params shape {} !"
                .format(key, value.shape, pre_value.shape))
    model.load_state_dict(new_state_dict, strict=False)


def save_model(model,
               optimizer,
               model_path,
               logger,
               is_best=False,
               prefix='ppocr',
               **kwargs):
    """
    save model to the target path
    """
    _mkdir_if_not_exist(model_path, logger)
    model_prefix = os.path.join(model_path, prefix)
    epoch = kwargs.get('epoch', 0)
    global_step = kwargs.get('global_step', 0)
    print("epoch: {}, global_step: {}".format(epoch, global_step))
    state_dict = {
        'epoch': epoch,
        'global_step': global_step,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state_dict, f"{model_prefix}.pth")

    # save metric and config
    if is_best:
        logger.info('save best model is to {}'.format(model_prefix))
    else:
        logger.info("save model in {}".format(model_prefix))
