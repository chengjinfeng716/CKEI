import torch
import torch.nn as nn
import numpy as np
import math

import src.utils.constant
from src.utils.common_layers import share_embedding, Encoder, Decoder, \
    Generator, LabelSmoothing, NoamOpt, CrossAttentionLayer

from sklearn.metrics import accuracy_score


def make_cross_attention_layer(args):
    params = (
        args.hidden_dim,
        args.total_key_depth,
        args.total_value_depth,
        args.filter_dim,
        args.num_heads,
        None,
        args.layer_dropout,
        args.attention_dropout,
        args.relu_dropout,
    )

    return CrossAttentionLayer(*params)


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        input_num = 5
        hid_num = 3
        if args.woIntent:
            input_num = input_num - 1
            hid_num = hid_num - 1
        if args.woNeed:
            input_num = input_num - 1
            hid_num = hid_num - 1
        if args.woWant:
            input_num = input_num - 1
            hid_num = hid_num - 1
        if args.woEffect:
            input_num = input_num - 1
            hid_num = hid_num - 1
        if args.woReact:
            input_num = input_num - 1
            hid_num = hid_num - 1

        if input_num == 0:
            return
        if hid_num < 1:
            hid_num = 1

        input_dim = input_num * args.hidden_dim
        hid_dim = hid_num * args.hidden_dim
        out_dim = args.hidden_dim

        self.lin_1 = nn.Linear(input_dim, hid_dim, bias=False)
        self.lin_2 = nn.Linear(hid_dim, out_dim, bias=False)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)

        return x


class CKEI(nn.Module):

    def __init__(self, args):
        super(CKEI, self).__init__()

        self.args = args
        self.embedding = share_embedding(args, args.vocab)

        # encoder for context sentences
        self.encoder = Encoder(args, args.emb_dim)

        if not self.args.woReact:
            # encoder for xReact sentences
            self.react_encoder = Encoder(args, args.emb_dim)

            # encoder for augmentation context outputs with xReact outputs
            self.react_aug_encoder = Encoder(args, 2 * args.emb_dim)

        if not self.args.woIntent or not self.args.woNeed or not self.args.woWant or not self.args.woEffect:
            # encoder for [xIntent, xNeed ,xWant, xEffect] sentences
            self.other_encoder = Encoder(args, args.emb_dim)

        # cross attention layer for augmentation context outputs with [xIntent, xNeed ,xWant, xEffect] outputs
        if not self.args.woIntent:
            self.intent_ca_layer = make_cross_attention_layer(args)
        if not self.args.woNeed:
            self.need_ca_layer = make_cross_attention_layer(args)
        if not self.args.woWant:
            self.want_ca_layer = make_cross_attention_layer(args)
        if not self.args.woEffect:
            self.effect_ca_layer = make_cross_attention_layer(args)

        if not self.args.woIntent or not self.args.woNeed or not self.args.woWant or not self.args.woEffect or not self.args.woReact:
            # mlp for integrate all augmentation encoder outputs
            self.mlp = MLP(args)

        # generator for emotional label
        self.emotion_lm_head = nn.Linear(args.hidden_dim, len(src.utils.constant.EMO_MAP), bias=False)

        if not self.args.woEmotionalIntensity:
            # emotional label word embedding vector
            self.emotion_latent = nn.Embedding(len(src.utils.constant.EMO_MAP), args.hidden_dim)

            # decoder used in the first decode stage
            self.vad_decoder = Decoder(args)

            # generator for predict the emotional intensity of the word to be generated
            self.vad_generator = nn.Linear(args.hidden_dim, 1)

            # loss function for emotional intensity
            self.vad_criterion = nn.MSELoss(reduction="sum")

        # decoder used in the second decode stage for the generated words in the response sentence
        self.decoder = Decoder(args)

        # generator for predict the word to be generated
        self.generator = Generator(args, args.vocab)

        # loss function
        self.criterion = LabelSmoothing(
            size=args.vocab.n_words,
            padding_idx=args.vocab.PAD_idx,
            smoothing=args.smoothing
        )

        # calculate ppl value
        self.criterion_ppl = nn.NLLLoss(ignore_index=args.vocab.PAD_idx)

        # optimizer
        self.optimizer = NoamOpt(
            args.hidden_dim,
            1,
            8000,
            torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
        )
        self.optimizer.optimizer.zero_grad()

    # augmentation the encoder outputs with commonsense context graph
    def forward(self, batch):
        context = batch["context_batch"]
        context_type = batch["context_type_batch"]

        x_intent = batch["x_intent_batch"]
        x_need = batch["x_need_batch"]
        x_want = batch["x_want_batch"]
        x_effect = batch["x_effect_batch"]
        x_react = batch["x_react_batch"]

        # encode the context sentences
        context_mask = context.eq(self.args.vocab.PAD_idx).unsqueeze(1)
        context_embedding = self.embedding(context) + self.embedding(context_type)
        context_outputs = self.encoder(context_embedding, context_mask)

        all_aug_outputs = []
        if not self.args.woIntent:
            # encode the x_intent sentences
            x_intent_mask = x_intent.eq(self.args.vocab.PAD_idx).unsqueeze(1)
            x_intent_embedding = self.embedding(x_intent)
            intent_outputs = self.other_encoder(x_intent_embedding, x_intent_mask)
            # augmentation the encoder outputs with x_intent outputs
            all_aug_outputs.append(self.intent_ca_layer(context_outputs, intent_outputs, x_intent_mask))

        if not self.args.woNeed:
            # encode the x_need sentences
            x_need_mask = x_need.eq(self.args.vocab.PAD_idx).unsqueeze(1)
            x_need_embedding = self.embedding(x_need)
            need_outputs = self.other_encoder(x_need_embedding, x_need_mask)
            # augmentation the encoder outputs with x_need outputs
            all_aug_outputs.append(self.need_ca_layer(context_outputs, need_outputs, x_need_mask))

        if not self.args.woWant:
            # encode the x_want sentences
            x_want_mask = x_want.eq(self.args.vocab.PAD_idx).unsqueeze(1)
            x_want_embedding = self.embedding(x_want)
            want_outputs = self.other_encoder(x_want_embedding, x_want_mask)
            # augmentation the encoder outputs with x_want outputs
            all_aug_outputs.append(self.want_ca_layer(context_outputs, want_outputs, x_want_mask))

        if not self.args.woEffect:
            # encode the x_effect sentences
            x_effect_mask = x_effect.eq(self.args.vocab.PAD_idx).unsqueeze(1)
            x_effect_embedding = self.embedding(x_effect)
            effect_outputs = self.other_encoder(x_effect_embedding, x_effect_mask)
            # augmentation the encoder outputs with x_effect outputs
            all_aug_outputs.append(self.effect_ca_layer(context_outputs, effect_outputs, x_effect_mask))

        if not self.args.woReact:
            # encode the x_react sentences
            x_react_mask = x_react.eq(self.args.vocab.PAD_idx).unsqueeze(1)
            x_react_embedding = self.embedding(x_react)
            x_react_outputs = self.react_encoder(x_react_embedding, x_react_mask)
            react_mean = torch.mean(x_react_outputs, dim=1).unsqueeze(1)
            dim = [-1, context_outputs.shape[1], -1]
            context_react_concat = torch.concat([context_outputs, react_mean.expand(dim)], dim=-1)
            # augmentation the encoder outputs with x_react outputs
            react_aug_output = self.react_aug_encoder(context_react_concat, context_mask)

            # Generate emotional label distribution
            emotion_logit = self.emotion_lm_head(react_aug_output[:, 0])

            all_aug_outputs.append(react_aug_output)
        else:
            emotion_logit = self.emotion_lm_head(context_outputs[:, 0])

        if all_aug_outputs:
            # integrate all augmentation encoder outputs
            all_aug_outputs = torch.cat(all_aug_outputs, dim=-1)
            ref_contrib = nn.Sigmoid()(all_aug_outputs)
            all_aug_outputs = ref_contrib * all_aug_outputs
            context_aug_outputs = self.mlp(all_aug_outputs)

        return context_mask, context_aug_outputs, emotion_logit

    def train_one_batch(self, batch, is_train, iter):
        context = batch["context_batch"]
        target = batch["target_batch"]
        target_vad = batch["target_vad_batch"]
        emotion = batch["emotion_batch"]
        target_vad_2 = batch["target_vad_batch2"]

        # encode context with commonsense knowledge
        context_mask, context_outputs, emotion_logit = self.forward(batch)

        # calculate the accuracy of emotional labels
        emotion_predict = np.argmax(emotion_logit.detach().cpu().numpy(), axis=1)
        emotion_acc = accuracy_score(emotion.cpu().numpy(), emotion_predict)

        # decode
        sos_token = torch.LongTensor([self.args.vocab.SOS_idx] * target.shape[0]) \
            .unsqueeze(1).to(self.args.device)

        target_shift = torch.cat((sos_token, target[:, :-1]), dim=1)
        target_mask = target_shift.eq(self.args.vocab.PAD_idx).unsqueeze(1)
        target_embedding = self.embedding(target_shift)

        if not self.args.woEmotionalIntensity:
            # the first decode stage
            target_outputs_vad, _ = self.vad_decoder(target_embedding, context_outputs, (context_mask, target_mask))

            # predict the emotional intensity of the word to be generated
            vad_logit = self.vad_generator(target_outputs_vad)
            vad_logit = torch.sigmoid(vad_logit)
            vad_logit_mask = 1 - target_mask.int()
            vad_logit_mask = vad_logit_mask.squeeze(1).unsqueeze(2)
            vad_logit = vad_logit * vad_logit_mask

            # whether to use the predict emotional intensity value or the ground truth
            r_tmp = np.random.rand()
            r_value = max(0.4, math.exp(-1. * iter / 20000))

            if r_value <= 0.4:
                print(iter)

            if r_tmp <= r_value:
                target_vad_softmax = target_vad.unsqueeze(2)
            else:
                target_vad_softmax = vad_logit

            # calculate the emotional vector corresponding to the word to be generated
            target_vad_softmax = target_vad_softmax.repeat(1, 1, context_outputs.size()[2])
            emotion_emb = self.emotion_latent(emotion).unsqueeze(1)
            emotion_emb = emotion_emb.repeat(1, target.shape[1], 1)
            emotion_emb = emotion_emb * target_vad_softmax

            # integrate the emotional vector
            target_embedding = target_embedding + emotion_emb

        # the second decode stage
        target_outputs, attn_dist = self.decoder(target_embedding, context_outputs, (context_mask, target_mask))

        # predict the word to be generated
        lm_logit = self.generator(target_outputs, attn_dist, context)

        if is_train:
            emotion_loss = nn.CrossEntropyLoss()(emotion_logit, emotion)
            word_loss = self.criterion(lm_logit.contiguous().view(-1, lm_logit.shape[-1]), target.contiguous().view(-1))

            if not self.args.woEmotionalIntensity:
                vad_loss = self.vad_criterion(vad_logit.contiguous().view(vad_logit.shape[0], vad_logit.shape[1]),
                                              target_vad_2)
                loss = 0.65 * emotion_loss + 0.25 * word_loss + 0.1 * vad_loss
            else:
                loss = 0.65 * emotion_loss + 0.25 * word_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer.optimizer.zero_grad()

        word_loss_ppl = self.criterion_ppl(lm_logit.contiguous().view(-1, lm_logit.shape[-1]),
                                           target.contiguous().view(-1)).item()
        return word_loss_ppl, math.exp(min(word_loss_ppl, 100)), emotion_acc

    def greedy_decoding(self, batch):

        context = batch["context_batch"]
        context_mask, context_outputs, emotion_logit = self.forward(batch)
        emotion_predict = torch.argmax(emotion_logit, dim=1)

        predict_output = torch.LongTensor([[self.args.vocab.PAD_idx]]) \
            .repeat(context.shape[0], self.args.max_dec_step).to(self.args.device)
        is_end = torch.zeros(context.shape[0], dtype=torch.bool).to(self.args.device)

        emotion_emb = self.emotion_latent(emotion_predict).unsqueeze(1)
        sos_token = torch.LongTensor([self.args.vocab.SOS_idx] * context.shape[0]).unsqueeze(1).to(self.args.device)
        for step in range(self.args.max_dec_step):

            target_shift = torch.cat((sos_token, predict_output[:, :step]), dim=1)
            target_mask = target_shift.eq(self.args.vocab.PAD_idx).unsqueeze(1)
            target_embedding = self.embedding(target_shift)

            if not self.args.woEmotionalIntensity:
                target_outputs_vad, _ = self.vad_decoder(target_embedding, context_outputs, (context_mask, target_mask))
                target_vad_softmax = self.vad_generator(target_outputs_vad)
                target_vad_softmax = torch.sigmoid(target_vad_softmax)

                target_vad_softmax = target_vad_softmax.repeat(1, 1, context_outputs.size()[2])
                emotion_emb_r = emotion_emb.repeat(1, target_shift.shape[1], 1)
                emotion_emb_r = emotion_emb_r * target_vad_softmax

                target_embedding = target_embedding + emotion_emb_r

            target_output, attn_dist = self.decoder(target_embedding, context_outputs, (context_mask, target_mask))
            lm_logit = self.generator(target_output, attn_dist, context)

            last_word_logit = lm_logit[:, -1]
            output = torch.argmax(last_word_logit, dim=-1)
            predict_output[:, step] = output

            predict_output[is_end, step] = self.args.vocab.PAD_idx
            is_end = is_end | output.eq(self.args.vocab.EOS_idx)
            if torch.sum(~is_end) == 0:
                break

        return predict_output, emotion_predict

