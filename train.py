import os
import torch
from torch.nn.init import xavier_uniform_
from tqdm import tqdm

from src.utils.common import setup_seed, setup_args, create_logger, make_infinite, valid_epoch, write_config
from src.utils.dataloader import prepare_dataloader
from src.model.ckei import CKEI


if __name__ == "__main__":
    setup_seed()
    args = setup_args()

    dataloader_train, dataloader_valid, dataloader_test, vocab_dict = prepare_dataloader(args)
    args.vocab = vocab_dict

    model = CKEI(args)
    model.to(args.device)
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight"):
            xavier_uniform_(p)

    train_logger = create_logger(args, "train")
    valid_logger = create_logger(args, "valid")
    write_config(args, train_logger)

    check_iter = 2000
    best_ppl = 1000
    patient = 0

    model.train()
    data_iter = make_infinite(dataloader_train)
    pbar = tqdm(range(1000000))
    for n_iter in pbar:

        loss, ppl, emotion_acc = model.train_one_batch(next(data_iter), True, n_iter)
        pbar.set_description("train loss {:.4f} ppl {:.4f} acc {:.4f}".format(loss, ppl, emotion_acc))
        train_logger.info("n_iter {} loss {} ppl {} emotion_acc {}".format((n_iter + 1), loss, ppl, emotion_acc))

        if (n_iter + 1) % check_iter == 0:
            print("\nstart valid")
            model.eval()
            val_loss, val_ppl, val_acc = valid_epoch(model, dataloader_valid, valid_logger)
            print("valid end with loss{:.4f} ppl {:.4f} acc {:.4f}".format(val_loss, val_ppl, val_acc))
            model.train()

            if n_iter < 12000:
                continue

            if val_ppl < best_ppl:
                patient = 0
                best_ppl = val_ppl
                train_logger.info("save best model for train n_iter {} best_ppl {}".format((n_iter + 1), best_ppl))
                torch.save(model.state_dict(), os.path.join(args.model_save_path,
                                                            "{}-{}-{:.4f}.model".format("best_model",
                                                                                        (n_iter + 1), best_ppl)))
            else:
                patient += 1

            if patient > 2:
                train_logger.info("early stopping")
                break
