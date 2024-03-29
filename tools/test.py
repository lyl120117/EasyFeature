from itertools import count
import os
import sys
import numpy as np
from tqdm import tqdm
import time
import platform

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))

from core.modeling.architectures import build_model
from core.data import build_dataloader
from core.utils.save_load import load_model
from core.utils.logging import get_logger
from core.postprocess import build_post_process
from core.metric import build_metric
from tools.program import ArgsParser, load_config, merge_config
from tools.visualize import save_hist, save_plots

import torch


def get_euc_hist(preds, labels):
    pred_idxs = preds.argmax(axis=1)
    positive_euc = [pred[label] for pred, label in zip(preds, labels)]
    positive_euc = np.array(positive_euc)
    negeative_euc = []
    for pred, label in zip(preds, labels):
        pred = np.delete(pred, label)
        negeative_euc.append(pred.max())
    negeative_euc = np.array(negeative_euc)
    error_positive_euc = positive_euc[pred_idxs != labels]
    error_negeative_euc = negeative_euc[pred_idxs != labels]
    return positive_euc, negeative_euc, error_positive_euc, error_negeative_euc


def eval(model, name, valid_dataloader, post_process_class, eval_class,
         device):
    model.eval()
    with torch.no_grad():
        total_frame = 0.0
        total_time = 0.0
        count = 0
        positive_eucs = []
        negeative_eucs = []
        error_positive_eucs = []
        error_negeative_eucs = []
        pbar = tqdm(total=len(valid_dataloader),
                    desc=name,
                    position=0,
                    leave=True)
        # max_iter = len(valid_dataloader) - 1 if platform.system(
        # ) == "Windows" else len(valid_dataloader)
        for idx, batch in enumerate(valid_dataloader):
            # if idx >= max_iter:
            #     break
            batch[0] = batch[0].to(device)
            batch[1] = batch[1].to(device)
            images = batch[0]
            start = time.time()
            preds = model(images)
            count += len(preds)
            batch = [item.cpu().numpy() for item in batch]
            # Obtain usable results from post-processing methods
            total_time += time.time() - start
            # Evaluate the results of the current batch

            post_result = post_process_class(preds, batch[1])
            eval_class(post_result, batch)

            positive_euc, negeative_euc, error_positive_euc, error_negeative_euc = get_euc_hist(
                preds.cpu().numpy(), batch[1])
            positive_eucs.extend(positive_euc)
            negeative_eucs.extend(negeative_euc)
            error_positive_eucs.extend(error_positive_euc)
            error_negeative_eucs.extend(error_negeative_euc)

            pbar.update(1)
            total_frame += len(images)
        # Get final metric，eg. acc or hmean
        metric = eval_class.get_metric()

    pbar.close()
    metric['count'] = int(count)
    metric['fps'] = int(total_frame / total_time)
    if len(positive_eucs) > 0:
        positive_eucs = np.array(positive_eucs)
        negeative_eucs = np.array(negeative_eucs)
        return [
            metric, positive_eucs, negeative_eucs, error_positive_eucs,
            error_negeative_eucs
        ]
    return metric


def log_str(metric):
    strs = []
    for k, v in metric.items():
        if type(v) is int:
            strs.append("{}: {}".format(k, v))
        elif type(v) is list:
            strs.append("{}: {}".format(k, v))
        else:
            strs.append('{}: {:x<6f}'.format(k, v))
    strs = ', '.join(strs)
    # 1 / 0
    return strs


def eval_feature(model,
                 name,
                 label_list,
                 template_mode,
                 template_dataloader,
                 valid_dataloader,
                 post_process_class,
                 eval_class,
                 device,
                 default_features=None):
    model.eval()
    with torch.no_grad():
        total_time = 0.0
        count = 0
        template_count = 0
        infer_template_count = 0
        templates = {}

        if template_mode == 1 or template_mode == 2:
            for i, default_feature in enumerate(default_features):
                label_key = label_list[i]
                default_feature = default_feature.reshape(1, -1)
                templates[label_key] = default_feature
                infer_template_count += 1

        if template_mode != 1:
            for batch in template_dataloader:
                start = time.time()
                batch[0] = batch[0].to(device)
                batch[1] = batch[1].to(device)
                images = batch[0]
                labels = batch[1]
                preds = model(images)

                preds = preds.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                for pred, label in zip(preds, labels):
                    pred = pred.reshape(1, -1)
                    label_key = label_list[label]
                    if label_key not in templates.keys():
                        templates[label_key] = pred
                        infer_template_count += 1
                    elif template_mode == 2:
                        templates[label_key] = np.append(templates[label_key],
                                                         pred,
                                                         axis=0)
                        infer_template_count += 1
                    template_count += 1
                # Obtain usable results from post-processing methods
                total_time += time.time() - start

        pbar = tqdm(total=len(valid_dataloader),
                    desc=name,
                    position=0,
                    leave=True)
        features = []
        # max_iter = len(valid_dataloader) - 1 if platform.system(
        # ) == "Windows" else len(valid_dataloader)
        pred_features = []
        for idx, batch in enumerate(valid_dataloader):
            # if idx >= max_iter:
            #     break
            start = time.time()
            batch[0] = batch[0].to(device)
            batch[1] = batch[1].to(device)
            images = batch[0]
            labels = batch[1]
            preds = model(images)
            pbar.update(1)
            for label, pred, img in zip(labels, preds, images):
                pred = pred.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                img = img.detach().cpu().numpy()

                features.append((pred, label, img))
                pred_features.append(pred)
                count += 1
            # Obtain usable results from post-processing methods
            total_time += time.time() - start

        pred_features = np.array(pred_features)
        post_result = post_process_class(templates, features)
        save(post_result, default_features, pred_features)
        eval_class(post_result)
        # Get final metric，eg. acc or hmean
        metric = eval_class.get_metric()

    pbar.close()
    metric['count'] = int(count)
    metric['template count'] = int(template_count)
    metric['inference template count'] = int(infer_template_count)
    metric['feature avg norms'] = np.mean(np.linalg.norm(pred_features,
                                                         axis=1))
    metric['fps'] = int(count / total_time)
    return metric


def save(args, default_features, features):
    cosines, inter_cosines, inner_cosines, inter_cosines_avg, inner_cosines_avg, pred_labels, true_labels = args

    positive_cosines = cosines[pred_labels == true_labels]
    negative_cosines = cosines[pred_labels != true_labels]
    root_dir = os.path.join(config['Global']['save_model_dir'], 'vis')
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    file_path = os.path.join(root_dir, 'positive_cosines.png')
    save_hist(file_path, positive_cosines)
    print('Saved figure to {}'.format(file_path))

    file_path = os.path.join(root_dir, 'negative_cosines.png')
    save_hist(file_path, negative_cosines)
    print('Saved figure to {}'.format(file_path))

    file_path = os.path.join(root_dir, 'inter_cosines.png')
    save_hist(file_path, inter_cosines)
    print('Saved figure to {}'.format(file_path))

    file_path = os.path.join(root_dir, 'inner_cosines.png')
    save_hist(file_path, inner_cosines)
    print('Saved figure to {}'.format(file_path))

    file_path = os.path.join(root_dir, 'inter_cosines_avg.png')
    save_hist(file_path, inter_cosines_avg)
    print('Saved figure to {}'.format(file_path))

    file_path = os.path.join(root_dir, 'inner_cosines_avg.png')
    save_hist(file_path, inner_cosines_avg)
    print('Saved figure to {}'.format(file_path))

    file_path = os.path.join(root_dir, 'fc_weight_norms.png')
    save_plots(file_path,
               np.linalg.norm(default_features, axis=1),
               format_strs='r.',
               labels='fc_weight_norms',
               xlabel='count',
               ylabel='fc_weight_norms')
    print('Saved figure to {}'.format(file_path))

    file_path = os.path.join(root_dir, 'feature_norms.png')
    save_plots(file_path,
               np.linalg.norm(features, axis=1),
               format_strs='r.',
               labels='feature_norms',
               xlabel='count',
               ylabel='feature_norms')
    print('Saved figure to {}'.format(file_path))


def main(config):
    global_config = config["Global"]
    device = ("cuda" if config["Global"]["use_gpu"]
              and torch.cuda.is_available() else "cpu")
    infer_feature = global_config.get("infer_feature", False)
    template_mode = global_config.get("template_mode", 0)

    log_file = os.path.join(global_config["save_model_dir"], "test.log")
    logger = get_logger(log_file=log_file)

    # build model
    model = build_model(config["Architecture"])
    load_model(logger, config, model)
    model = model.to(device)
    model.eval()

    ps = getattr(model.head, 'ps', None)
    if ps is not None:
        ps = ps.detach().cpu().numpy()
        logger.info("ps: {}".format(ps))

    # build metric
    eval_class = build_metric(config["Metric"])
    # build post process
    post_process_class = build_post_process(config["PostProcess"],
                                            global_config)
    label_list = post_process_class.label_list
    logger.info(f"Dictionary size: {len(label_list)}")
    if len(label_list) < 100:
        logger.info(f"Dictionary list: {label_list}")

    test_dataloader = build_dataloader(config, "Test", logger)

    test_metric = eval(model, "Test", test_dataloader, post_process_class,
                       eval_class, device)

    if type(test_metric) is list:
        test_metric, positive_eucs, negeative_eucs, error_positive_eucs, error_negeative_eucs = test_metric
        root_dir = os.path.join(config['Global']['save_model_dir'], 'vis')
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        file_path = os.path.join(root_dir, 'inner_eucs.png')
        save_plots(file_path, positive_eucs, 'g.', 'positive_eucs')
        print('Saved figure to {}'.format(file_path))

        file_path = os.path.join(root_dir, 'inter_eucs.png')
        save_plots(file_path, negeative_eucs, 'r.', 'negeative_eucs')
        print('Saved figure to {}'.format(file_path))

        file_path = os.path.join(root_dir, 'error_eucs.png')
        save_plots(file_path, [error_positive_eucs, error_negeative_eucs],
                   ['g.', 'r.'], ['positive', 'negative'])
        print('Saved figure to {}'.format(file_path))

    logger.info(log_str(test_metric))
    if infer_feature:
        backbone = model.backbone
        template_dataloader = build_dataloader(config, "Template", logger)
        test_dataloader = build_dataloader(config, "TestTemplate", logger)
        feature_eval_class = build_metric(config["FeatureMetric"])
        post_feature_process_class = build_post_process(
            config["PostFeatureProcess"], global_config)

        default_features = model.head.get_weights().detach().cpu().numpy()
        default_feature_norms = np.linalg.norm(default_features, axis=1)
        logger.info("FC weight avg norm: {}".format(
            np.mean(default_feature_norms)))
        if template_mode == 1:
            logger.info(
                "template mode is 1, use fc weight as default features")
        elif template_mode == 2:
            logger.info(
                "template mode is 2, combine fc weight and template features")
        else:
            logger.info("template mode is 0, use template features")

        feature_metirc = eval_feature(backbone,
                                      "Cal Feature",
                                      label_list,
                                      template_mode,
                                      template_dataloader,
                                      test_dataloader,
                                      post_feature_process_class,
                                      feature_eval_class,
                                      device,
                                      default_features=default_features)
        logger.info(log_str(feature_metirc))


if __name__ == "__main__":
    FLAGS = ArgsParser().parse_args()
    profiler_options = FLAGS.profiler_options
    config = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    main(config)
