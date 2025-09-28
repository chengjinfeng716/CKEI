import torch
import math
import numpy as np
import logging
import os
import datetime
from argparse import ArgumentParser
from tqdm import tqdm


def setup_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_args():

    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/dataset_preproc.p")
    parser.add_argument("--vad_dir", type=str, default="data/VAD.json")
    parser.add_argument("--emb_file", type=str, default="data/glove.6B.300d.txt")
    parser.add_argument("--best_mode_file", type=str, default="")
    parser.add_argument("--result_path", type=str, default="result")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--smoothing", type=float, default=0.1)

    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=300)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--total_key_depth", type=int, default=40)
    parser.add_argument("--total_value_depth", type=int, default=40)
    parser.add_argument("--filter_dim", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=1000)
    parser.add_argument("--input_dropout", type=float, default=0.0)
    parser.add_argument("--layer_dropout", type=float, default=0.0)
    parser.add_argument("--attention_dropout", type=float, default=0.0)
    parser.add_argument("--relu_dropout", type=float, default=0.0)

    parser.add_argument("--pointer_gen", action="store_true")
    parser.add_argument("--max_dec_step", type=int, default=30)

    parser.add_argument("--woIntent", action="store_true")
    parser.add_argument("--woNeed", action="store_true")
    parser.add_argument("--woWant", action="store_true")
    parser.add_argument("--woEffect", action="store_true")
    parser.add_argument("--woReact", action="store_true")
    parser.add_argument("--woEmotionalIntensity", action="store_true")

    args = parser.parse_args()
    result_path = args.result_path
    if args.woIntent:
        result_path = result_path + "_woIntent"
    if args.woNeed:
        result_path = result_path + "_woNeed"
    if args.woWant:
        result_path = result_path + "_woWant"
    if args.woEffect:
        result_path = result_path + "_woEffect"
    if args.woReact:
        result_path = result_path + "_woReact"
    if args.woEmotionalIntensity:
        result_path = result_path + "_woEmotionalIntensity"

    model_save_path = os.path.join(result_path, "model")
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    args.model_save_path = model_save_path

    log_save_path = os.path.join(result_path, "log")
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)
    args.log_save_path = log_save_path

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args


def create_logger(args, name_prefix):
    logger = logging.getLogger(name_prefix)
    logger.setLevel(logging.INFO)

    log_file_path = os.path.join(args.log_save_path,
                                 "{}-{}.log".format(name_prefix, datetime.datetime.today().strftime('%Y%m%d%H%M%S')))
    file_handler = logging.FileHandler(filename=log_file_path, encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


def valid_epoch(model, dataloader, logger):

    t_loss = []
    t_ppl = []
    t_emotion_acc = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for idx, batch in pbar:
            loss, ppl, emotion_acc = model.train_one_batch(batch, False, 0)
            logger.info("idx {} loss {} ppl {} emotion_acc {}".format((idx + 1), loss, ppl, emotion_acc))

            t_loss.append(loss)
            t_ppl.append(ppl)
            t_emotion_acc.append(emotion_acc)

    loss_mean = np.mean(t_loss)
    ppl_mean = np.mean(t_ppl)
    emotion_acc_mean = np.mean(t_emotion_acc)

    logger.info("loss_mean {} ppl_mean {} emotion_acc_mean {}".format(loss_mean, math.exp(loss_mean), emotion_acc_mean))
    return loss_mean, math.exp(loss_mean), emotion_acc_mean


def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x


def print_sample(emotion, hyp_emotion, context, target, hyp):
    res = ""
    res += "emotion: {}".format(emotion) + "\n"
    res += "hyp_emotion: {}".format(hyp_emotion) + "\n"
    res += "context: {}".format(context) + "\n"
    res += "target: {}".format(target) + "\n"
    res += "hyp: {}".format(hyp) + "\n"
    res += "---------------------------------------------------------------" + "\n"
    return res


def write_config(args, logger):
    logger.info("------------ config set -------------")
    for k, v in args.__dict__.items():
        if "False" in str(v):
            pass
        elif "True" in str(v):
            logger.info("--{} ".format(k))
        else:
            logger.info("--{}={} ".format(k, v))
    logger.info("------------ config set -------------")
