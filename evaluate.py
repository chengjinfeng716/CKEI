
import torch
from tqdm import tqdm

from src.utils.common import setup_seed, setup_args, create_logger, valid_epoch, print_sample, write_config
from src.utils.dataloader import prepare_dataloader
from src.model.ckei import CKEI
from src.utils.constant import MAP_EMO as map_emo
from nltk import ngrams
from collections import Counter
from nltk import word_tokenize


def distinct_metric(hypothesis):

    unigram_counter, bigram_counter = Counter(), Counter()
    for hypo in hypothesis:
        tokens = word_tokenize(hypo)
        unigram_counter.update(tokens)
        bigram_counter.update(ngrams(tokens, 2))

    distinct_1 = len(unigram_counter) / sum(unigram_counter.values())
    distinct_2 = len(bigram_counter) / sum(bigram_counter.values())
    return distinct_1, distinct_2


if __name__ == "__main__":
    setup_seed()
    args = setup_args()

    _, _, dataloader_test, vocab_dict = prepare_dataloader(args)
    args.vocab = vocab_dict

    model = CKEI(args)
    model.to(args.device)

    previous_state = torch.load(args.best_mode_file)
    model.load_state_dict(previous_state)
    model.eval()

    logger = create_logger(args, "evaluate")
    write_config(args, logger)

    loss, ppl, emotion_acc = valid_epoch(model, dataloader_test, logger)

    hypothesis = []
    references = []
    generate_sample = []

    pbar = tqdm(enumerate(dataloader_test), total=len(dataloader_test))
    for _, batch in pbar:
        predict_output, emotion_predict = model.greedy_decoding(batch)
        for i, hyp_ids in enumerate(predict_output):
            hyp = args.vocab.vect2text(hyp_ids.tolist())
            eos_idx = hyp.find(args.vocab.eos_token)
            if eos_idx != -1:
                hyp = hyp[:eos_idx]
            hypothesis.append(hyp)
            hyp_emotion = map_emo[emotion_predict[i].item()]

            context_text = [" ".join(s) for s in batch["context_text"][i]]
            target_text = " ".join(batch["target_text"][i])
            references.append(target_text)

            generate_sample.append(print_sample(batch["emotion_text"][i],
                                                hyp_emotion,
                                                context_text,
                                                target_text,
                                                hyp))

    dist_1, dist_2 = distinct_metric(hypothesis)

    summary_logger = create_logger(args, "evaluate_summary")
    summary_logger.info(model)
    write_config(args, summary_logger)

    print("evaluate ppl {} emotion_acc {} dist-1 {} dist-2 {}".format(ppl, emotion_acc, dist_1, dist_2))
    summary_logger.info("evaluate ppl {} emotion_acc {} dist-1 {} dist-2 {}".format(ppl, emotion_acc, dist_1, dist_2))

    for sample in generate_sample:
        summary_logger.info(sample)
