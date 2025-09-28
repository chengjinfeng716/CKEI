import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """

    def __init__(
        self,
        input_depth,
        total_key_depth,
        total_value_depth,
        output_depth,
        num_heads,
        bias_mask=None,
        dropout=0.0,
    ):
        """
        Parameters:
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadAttention, self).__init__()
        # Checks borrowed from
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py

        if total_key_depth % num_heads != 0:
            print(
                "Key depth (%d) must be divisible by the number of "
                "attention heads (%d)." % (total_key_depth, num_heads)
            )
            total_key_depth = total_key_depth - (total_key_depth % num_heads)
        if total_value_depth % num_heads != 0:
            print(
                "Value depth (%d) must be divisible by the number of "
                "attention heads (%d)." % (total_value_depth, num_heads)
            )
            total_value_depth = total_value_depth - (total_value_depth % num_heads)

        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5  ## sqrt
        self.bias_mask = bias_mask

        # Key and query depth will be same
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(
            shape[0], shape[1], self.num_heads, shape[2] // self.num_heads
        ).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (
            x.permute(0, 2, 1, 3)
            .contiguous()
            .view(shape[0], shape[2], shape[3] * self.num_heads)
        )

    def forward(self, queries, keys, values, mask):

        # Do a linear for each component
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)

        # Split into multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # Scale queries
        queries *= self.query_scale

        # Combine queries and keys
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            logits = logits.masked_fill(mask, -1e18)

        # attention weights
        attention_weights = logits.sum(dim=1) / self.num_heads

        # Convert to probabilites
        weights = nn.functional.softmax(logits, dim=-1)

        # Dropout
        weights = self.dropout(weights)

        # Combine with values to get context
        contexts = torch.matmul(weights, values)

        # Merge heads
        contexts = self._merge_heads(contexts)

        # Linear to get output
        outputs = self.output_linear(contexts)

        return outputs, attention_weights


class Conv(nn.Module):
    """
    Convenience class that does padding and convolution for inputs in the format
    [batch_size, sequence length, hidden size]
    """

    def __init__(self, input_size, output_size, kernel_size, pad_type):
        """
        Parameters:
            input_size: Input feature size
            output_size: Output feature size
            kernel_size: Kernel width
            pad_type: left -> pad on the left side (to mask future data),
                      both -> pad on both sides
        """
        super(Conv, self).__init__()
        padding = (
            (kernel_size - 1, 0)
            if pad_type == "left"
            else (kernel_size // 2, (kernel_size - 1) // 2)
        )
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(
            input_size, output_size, kernel_size=kernel_size, padding=0
        )

    def forward(self, inputs):
        inputs = self.pad(inputs.permute(0, 2, 1))
        outputs = self.conv(inputs).permute(0, 2, 1)

        return outputs


class PositionwiseFeedForward(nn.Module):
    """
    Does a Linear + RELU + Linear on each of the timesteps
    """

    def __init__(
        self,
        input_depth,
        filter_size,
        output_depth,
        layer_config="ll",
        padding="left",
        dropout=0.0,
    ):
        """
        Parameters:
            input_depth: Size of last dimension of input
            filter_size: Hidden size of the middle layer
            output_depth: Size last dimension of the final output
            layer_config: ll -> linear + ReLU + linear
                          cc -> conv + ReLU + conv etc.
            padding: left -> pad on the left side (to mask future data),
                     both -> pad on both sides
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(PositionwiseFeedForward, self).__init__()

        layers = []
        sizes = (
            [(input_depth, filter_size)]
            + [(filter_size, filter_size)] * (len(layer_config) - 2)
            + [(filter_size, output_depth)]
        )

        for lc, s in zip(list(layer_config), sizes):
            if lc == "l":
                layers.append(nn.Linear(*s))
            elif lc == "c":
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError("Unknown layer type {}".format(lc))

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)

        return x


class LayerNorm(nn.Module):
    # Borrowed from jekbradbury
    # https://github.com/pytorch/pytorch/issues/1959
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class EncoderLayer(nn.Module):
    """
    Represents one Encoder layer of the Transformer Encoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(
        self,
        hidden_size,
        total_key_depth,
        total_value_depth,
        filter_size,
        num_heads,
        bias_mask=None,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
    ):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(
            hidden_size,
            total_key_depth,
            total_value_depth,
            hidden_size,
            num_heads,
            bias_mask,
            attention_dropout,
        )

        self.positionwise_feed_forward = PositionwiseFeedForward(
            hidden_size,
            filter_size,
            hidden_size,
            layer_config="cc",
            padding="both",
            dropout=relu_dropout,
        )
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs, mask=None):
        x = inputs

        # Layer Normalization
        x_norm = self.layer_norm_mha(x)

        # Multi-head attention
        y, _ = self.multi_head_attention(x_norm, x_norm, x_norm, mask)

        # Dropout and residual
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual
        y = self.dropout(x + y)

        return y


class DecoderLayer(nn.Module):
    """
    Represents one Decoder layer of the Transformer Decoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(
        self,
        hidden_size,
        total_key_depth,
        total_value_depth,
        filter_size,
        num_heads,
        bias_mask,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
    ):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(DecoderLayer, self).__init__()

        self.multi_head_attention_dec = MultiHeadAttention(
            hidden_size,
            total_key_depth,
            total_value_depth,
            hidden_size,
            num_heads,
            bias_mask,
            attention_dropout,
        )

        self.multi_head_attention_enc_dec = MultiHeadAttention(
            hidden_size,
            total_key_depth,
            total_value_depth,
            hidden_size,
            num_heads,
            None,
            attention_dropout,
        )

        self.positionwise_feed_forward = PositionwiseFeedForward(
            hidden_size,
            filter_size,
            hidden_size,
            layer_config="cc",
            padding="left",
            dropout=relu_dropout,
        )
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha_dec = LayerNorm(hidden_size)
        self.layer_norm_mha_enc = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        """
        NOTE: Inputs is a tuple consisting of decoder inputs and encoder output
        """

        x, encoder_outputs, attention_weight, mask = inputs
        mask_src, dec_mask = mask

        # Layer Normalization before decoder self attention
        x_norm = self.layer_norm_mha_dec(x)

        # Masked Multi-head attention
        y, _ = self.multi_head_attention_dec(x_norm, x_norm, x_norm, dec_mask)

        # Dropout and residual after self-attention
        x = self.dropout(x + y)

        # Layer Normalization before encoder-decoder attention
        x_norm = self.layer_norm_mha_enc(x)

        # Multi-head encoder-decoder attention
        y, attention_weight = self.multi_head_attention_enc_dec(
            x_norm, encoder_outputs, encoder_outputs, mask_src
        )

        # Dropout and residual after encoder-decoder attention
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual after positionwise feed forward layer
        y = self.dropout(x + y)

        # Return encoder outputs as well to work with nn.Sequential
        return y, encoder_outputs, attention_weight, mask


class CrossAttentionLayer(nn.Module):
    """
    Represents one cross attention layer
    """
    def __init__(
        self,
        hidden_size,
        total_key_depth,
        total_value_depth,
        filter_size,
        num_heads,
        bias_mask=None,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
    ):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(CrossAttentionLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(
            hidden_size,
            total_key_depth,
            total_value_depth,
            hidden_size,
            num_heads,
            bias_mask,
            attention_dropout,
        )

        self.positionwise_feed_forward = PositionwiseFeedForward(
            hidden_size,
            filter_size,
            hidden_size,
            layer_config="cc",
            padding="both",
            dropout=relu_dropout,
        )
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, outputs, attention_outputs, attention_mask):
        x = outputs

        # Layer Normalization
        x_norm = self.layer_norm_mha(x)

        # Multi-head attention
        y, _ = self.multi_head_attention(x_norm, attention_outputs, attention_outputs, attention_mask)

        # Dropout and residual
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual
        y = self.dropout(x + y)

        return y


class Encoder(nn.Module):

    def __init__(self, args, emb_dim):
        super(Encoder, self).__init__()

        self.args = args
        self.num_layers = args.num_layers
        self.timing_signal = gen_timing_signal(args.max_length, args.hidden_dim)

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

        self.embedding_proj = nn.Linear(emb_dim, args.hidden_dim, bias=False)
        self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(args.num_layers)])

        self.layer_norm = LayerNorm(args.hidden_dim)
        self.input_dropout = nn.Dropout(args.input_dropout)

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        # Add timing signal
        x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

        for i in range(self.num_layers):
            x = self.enc[i](x, mask)

        y = self.layer_norm(x)
        return y


class Decoder(nn.Module):

    def __init__(self, args):
        super(Decoder, self).__init__()

        self.args = args
        self.num_layers = args.num_layers
        self.timing_signal = gen_timing_signal(args.max_length, args.hidden_dim)
        self.mask = get_attn_subsequent_mask(args.max_length).to(self.args.device)

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

        self.dec = nn.Sequential(
            *[DecoderLayer(*params) for l in range(args.num_layers)]
        )

        self.embedding_proj = nn.Linear(args.emb_dim, args.hidden_dim, bias=False)
        self.layer_norm = LayerNorm(args.hidden_dim)
        self.input_dropout = nn.Dropout(args.input_dropout)

    def forward(self, inputs, encoder_output, mask):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(
            mask_trg + self.mask[:, : mask_trg.size(-1), : mask_trg.size(-1)], 0
        )
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        # Add timing signal
        x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

        # Run decoder
        y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))

        # Final layer normalization
        y = self.layer_norm(y)

        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, args, vocab):
        super(Generator, self).__init__()
        self.args = args
        self.proj = nn.Linear(args.hidden_dim, vocab.n_words)
        self.p_gen_linear = nn.Linear(args.hidden_dim, 1)

    def forward(
        self,
        x,
        attn_dist=None,
        enc_batch_extend_vocab=None,
        temp=1
    ):

        logit = self.proj(x)

        if self.args.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist / temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat(
                [enc_batch_extend_vocab.unsqueeze(1)] * x.size(1), 1
            )

            logit = torch.log(
                vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_)
            )
            return logit
        else:
            return F.log_softmax(logit, dim=-1)


class LabelSmoothing(nn.Module):

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.size()[0] > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def state_dict(self):
        return self.optimizer.state_dict()

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )


def gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        float(num_timescales) - 1
    )
    inv_timescales = min_timescale * np.exp(
        np.arange(num_timescales).astype(np.float) * -log_timescale_increment
    )
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(
        signal, [[0, 0], [0, channels % 2]], "constant", constant_values=[0.0, 0.0]
    )
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)


def get_attn_subsequent_mask(max_length):
    """
    Get an attention mask to avoid using the subsequent info.
    Args:
        max_length: int
    Returns:
        (`LongTensor`):
        * subsequent_mask `[1 x size x size]`
    """
    attn_shape = (1, max_length, max_length)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    subsequent_mask = torch.from_numpy(subsequent_mask)

    return subsequent_mask


def gen_embeddings(args, vocab):
    """
    Generate an initial embedding matrix for `word_dict`.
    If an embedding file is not given or a word is not in the embedding file,
    a randomly initialized vector will be used.
    """
    embeddings = np.random.randn(vocab.n_words, args.emb_dim) * 0.01
    print("Embeddings: %d x %d" % (vocab.n_words, args.emb_dim))
    if args.emb_file is not None:
        print("Loading embedding file: %s" % args.emb_file)
        pre_trained = 0
        for line in open(args.emb_file, mode="r", encoding="utf-8").readlines():
            sp = line.split()
            if len(sp) == args.emb_dim + 1:
                if sp[0] in vocab.word2index:
                    pre_trained += 1
                    embeddings[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]]
            else:
                print(sp[0])
        print(
            "Pre-trained: %d (%.2f%%)"
            % (pre_trained, pre_trained * 100.0 / vocab.n_words)
        )
    return embeddings


class Embeddings(nn.Module):
    def __init__(self, n_words, embedding_size, padding_idx=None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_words, embedding_size, padding_idx=padding_idx)
        self.embedding_size = embedding_size

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.embedding_size)


def share_embedding(args, vocab):
    embedding = Embeddings(vocab.n_words, args.emb_dim, padding_idx=vocab.PAD_idx)

    pre_embedding = gen_embeddings(args, vocab)
    embedding.lut.weight.data.copy_(torch.FloatTensor(pre_embedding))
    embedding.lut.weight.data.requires_grad = True

    return embedding

