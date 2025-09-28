import os
import torch
import pickle
import json

from torch.nn.utils.rnn import pad_sequence
from src.utils.constant import EMO_MAP as emo_map
from src.utils.vocab_dictionary import VocabDictionary


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data, vocab, args):
        self.data = data
        self.vocab = vocab
        self.args = args

    def __len__(self):
        return len(self.data["emotion"])

    def __getitem__(self, index):

        item = {}
        item["context_text"] = self.data["context"][index]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotion"][index]

        item["x_intent_text"] = self.data["utt_cs"][index][0]
        item["x_need_text"] = self.data["utt_cs"][index][1]
        item["x_want_text"] = self.data["utt_cs"][index][2]
        item["x_effect_text"] = self.data["utt_cs"][index][3]
        item["x_react_text"] = self.data["utt_cs"][index][4]

        item["context"], item["context_type"] = self.process_context(item["context_text"])
        item["target"], item["target_vad"] = self.process_target(item["target_text"])
        item["emotion"] = emo_map[item["emotion_text"]]

        item["x_intent"] = self.process_cs(item["x_intent_text"])
        item["x_need"] = self.process_cs(item["x_need_text"])
        item["x_want"] = self.process_cs(item["x_want_text"])
        item["x_effect"] = self.process_cs(item["x_effect_text"])
        item["x_react"] = self.process_cs(item["x_react_text"], True)

        return item

    def process_context(self, context_text):

        context = [self.vocab.CLS_idx]
        context_type = [self.vocab.CLS_idx]

        for i, sentence in enumerate(context_text):
            context += self.vocab.text2vect(sentence)
            spk = self.vocab.USR_idx if i % 2 == 0 else self.vocab.SYS_idx
            context_type += [spk for _ in range(len(sentence))]

        assert len(context) == len(context_type)
        return torch.LongTensor(context), torch.LongTensor(context_type)

    def process_target(self, target_text):
        target = self.vocab.text2vect(target_text)
        target += [self.vocab.EOS_idx]

        target_vad = self.vocab.vad_sentence(target_text, 0) + [0]

        return torch.LongTensor(target), torch.FloatTensor(target_vad)

    def process_cs(self, cs_text, react=False):

        sequence = [] if react else [self.vocab.CLS_idx]
        for sentence in cs_text:
            sequence += self.vocab.text2vect(sentence)
        return torch.LongTensor(sequence)

    def collate(self, items):

        items.sort(key=lambda x: len(x["context"]), reverse=True)

        item_info = {}
        for key in items[0].keys():
            item_info[key] = [item[key] for item in items]

        context_batch = pad_sequence(item_info["context"], batch_first=True, padding_value=self.vocab.PAD_idx)
        context_type_batch = pad_sequence(item_info["context_type"], batch_first=True, padding_value=self.vocab.PAD_idx)
        target_batch = pad_sequence(item_info["target"], batch_first=True, padding_value=self.vocab.PAD_idx)
        target_vad_batch = pad_sequence(item_info["target_vad"], batch_first=True, padding_value=0)
        emotion_batch = torch.LongTensor(item_info["emotion"])

        x_intent_batch = pad_sequence(item_info["x_intent"], batch_first=True, padding_value=self.vocab.PAD_idx)
        x_need_batch = pad_sequence(item_info["x_need"], batch_first=True, padding_value=self.vocab.PAD_idx)
        x_want_batch = pad_sequence(item_info["x_want"], batch_first=True, padding_value=self.vocab.PAD_idx)
        x_effect_batch = pad_sequence(item_info["x_effect"], batch_first=True, padding_value=self.vocab.PAD_idx)
        x_react_batch = pad_sequence(item_info["x_react"], batch_first=True, padding_value=self.vocab.PAD_idx)

        batch = {}
        batch["context_batch"] = context_batch.to(self.args.device)
        batch["context_type_batch"] = context_type_batch.to(self.args.device)
        batch["target_batch"] = target_batch.to(self.args.device)
        batch["target_vad_batch"] = target_vad_batch.to(self.args.device)
        batch["target_vad_batch2"] = target_vad_batch.to(self.args.device)
        batch["emotion_batch"] = emotion_batch.to(self.args.device)

        batch["x_intent_batch"] = x_intent_batch.to(self.args.device)
        batch["x_need_batch"] = x_need_batch.to(self.args.device)
        batch["x_want_batch"] = x_want_batch.to(self.args.device)
        batch["x_effect_batch"] = x_effect_batch.to(self.args.device)
        batch["x_react_batch"] = x_react_batch.to(self.args.device)

        batch["context_text"] = item_info["context_text"]
        batch["target_text"] = item_info["target_text"]
        batch["emotion_text"] = item_info["emotion_text"]

        batch["x_intent_text"] = item_info["x_intent_text"]
        batch["x_need_text"] = item_info["x_need_text"]
        batch["x_want_text"] = item_info["x_want_text"]
        batch["x_effect_text"] = item_info["x_effect_text"]
        batch["x_react_text"] = item_info["x_react_text"]

        return batch


def prepare_dataloader(args):

    if os.path.exists(args.data_dir):
        with open(args.data_dir, "rb") as f:
            [data_tra, data_val, data_tst, lang] = pickle.load(f)
    else:
        raise ValueError("data file is not exist")

    vocab_dict = VocabDictionary()
    vocab_dict.word2index = lang.word2index
    vocab_dict.word2count = lang.word2count
    vocab_dict.index2word = lang.index2word
    vocab_dict.n_words = lang.n_words

    with open(args.vad_dir, "r", encoding="utf-8") as f:
        vad = json.load(f)
    vocab_dict.vad = vad

    dataset_train = Dataset(data_tra, vocab_dict, args)
    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   collate_fn=dataset_train.collate)

    dataset_valid = Dataset(data_val, vocab_dict, args)
    dataloader_valid = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   collate_fn=dataset_valid.collate)

    dataset_test = Dataset(data_tst, vocab_dict, args)
    dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  collate_fn=dataset_test.collate)

    return dataloader_train, dataloader_valid, dataloader_test, vocab_dict

