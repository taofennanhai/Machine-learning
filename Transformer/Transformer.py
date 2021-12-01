import torch
import torch.nn as nn

from Decoder import Decoder
from Encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, d_model=512, target_vocab_size=9, device='cuda'):
        super(Transformer, self).__init__()
        self.device = device
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)
        self.projection = nn.Linear(d_model, target_vocab_size, bias=False).to(device)    # 前向传播层，输出每个词的概率

    def forward(self, encoder_inputs, decoder_inputs):
        """Transformers的输入：两个序列
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        """
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # encoder_outputs: [batch_size, src_len, d_model], encoder_self_attentions: [n_layers, batch_size, n_heads, src_len, src_len]
        # 经过Encoder网络后，得到的输出还是[batch_size, src_len, d_model]
        encoder_outputs, encoder_self_attentions = self.encoder(encoder_inputs)    # 经过encoder层输出，后面那个输入主要用于热力图

        # decoder_outputs: [batch_size, tgt_len, d_model], decoder_self_attentions: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        decoder_outputs, decoder_self_attentions, decoder_encoder_attentions = self.decoder(decoder_inputs, encoder_inputs, encoder_outputs)    # decoder输出可以

        # decoder_outputs: [batch_size, tgt_len, d_model] -> decoder_logits: [batch_size, tgt_len, tgt_vocab_size]
        decoder_logits = self.projection(decoder_outputs)    # 进行单词的预测

        # View把每批句子拉成一条长句子，这样方便进行交叉熵损计算   decoder_logits: [batch_size, tgt_len, tgt_vocab_size] -> [batch_size*tgt_len, tgt_vocab_size]
        return decoder_logits.view(-1, decoder_logits.size(-1)), encoder_self_attentions, decoder_self_attentions, decoder_encoder_attentions