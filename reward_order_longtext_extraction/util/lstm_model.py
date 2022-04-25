import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from argparse import ArgumentParser
import reward_order_longtext_extraction.util.ner_util as ner_util

# from allennlp.modules import ConditionalRandomField
from torchcrf import CRF
import numpy as np


class NerRNN(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_layers", type=int, default=1)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--weight_decay", type=float, default=0.001)
        parser.add_argument("--adam_eps", type=float, default=1e-6)
        parser.add_argument("--adam_beta1", type=float, default=0.95)
        parser.add_argument("--adam_beta2", type=float, default=0.99)
        parser.add_argument("--rnn_type", type=str, default="lstm")
        parser.add_argument("--use_crf", type=bool, default=True)
        parser.add_argument("--layer_lr", type=bool, default=False)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--hidden_size", type=int, default=128)
        parser.add_argument("--embedding_size", type=int, default=768)
        return parser

    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tag_map = {"PAD": 0, "B": 1, "I": 2, "E": 3, "O": 4}
        self.num_classes = num_classes  # B I E O   4类
        self.learning_rate = self.hparams.learning_rate
        self.crf_learning_rate = 0.001
        assert self.hparams.rnn_type in ["lstm", "qrnn", "cnn"], "RNN type is not supported"
        self.dropout = nn.Dropout(self.hparams.dropout)
        if self.hparams.rnn_type == "lstm":
            self.lstm = nn.LSTM(
                self.hparams.embedding_size,
                self.hparams.hidden_size,
                self.hparams.num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=self.hparams.dropout,
            )

        if self.hparams.use_crf:
            self.linear = nn.Linear(self.hparams.hidden_size * 2, self.num_classes)
            self.crf =CRF(num_tags=self.num_classes, batch_first=True)
        else:
            self.linear = nn.Linear(self.hparams.hidden_size * 2, self.num_classes)

        #self.f1_metric = pl.metrics.F1(num_classes=self.num_classes)

    def forward(self, x):
        # Set initial states
        # emb = self.embedding(x)
        # emb = self.dropout(emb)
        if self.hparams.rnn_type == "lstm":
            output, self.hidden = self.lstm(x)

            # output2 = torch.cat((output,x),2)

            # print(output.size())
            # print(x.size())
            # print(output2.size())

        if self.hparams.use_crf:
            x_mask = x.mean(axis=2).eq(0).eq(0)  # x.eq(0).eq(0)
            logits= self.linear(output)
            y_pred = torch.FloatTensor(self.crf.decode(logits))
        else:
            logits= self.linear(output)
            y_pred = F.softmax(logits, dim=2)

        return (y_pred,logits)

    def training_step(self, batch, batch_idx):
        embed, tags = batch

        outputs,logits = self.forward(embed)
        if self.hparams.use_crf:
            mask_sent = embed.mean(axis=2).eq(0).eq(0)  # mask_sent = sent.eq(0).eq(0)
            # loss = self.crf.forward(outputs, tags, mask_sent)
            # loss = self.crf.forward(outputs, tags, mask_sent)
            loss = -self.crf(logits,tags,mask_sent)
        else:
            loss = F.cross_entropy(torch.flatten(outputs, 0, 1), torch.flatten(tags, 0, 1), ignore_index=0)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, data_batch, batch_nb):
        sent, tags = data_batch
        outputs,logits = self.forward(sent)
        if self.hparams.use_crf:
            mask_sent = sent.mean(axis=2).eq(0).eq(0)  # mask_sent = sent.eq(0).eq(0)
            loss_val = -self.crf(logits,tags,mask_sent)
        else:
            loss_val = F.cross_entropy(torch.flatten(outputs, 0, 1), torch.flatten(tags, 0, 1), ignore_index=0)
        # mask = tags.eq(0).eq(0)
        predicted = outputs.argmax(2) if not self.hparams.use_crf else outputs

        self.log("val_loss", loss_val)
        return loss_val, predicted, tags

    def validation_epoch_end(self, epoch_outputs):
        f1_list = []
        p_list = []
        r_list = []
        loss_list = []
        predicted_list = None
        tag_list = None

        for loss, predicted, tags in epoch_outputs:
            if predicted_list is None:
                predicted_list = predicted.flatten()
            predicted_list = torch.cat((predicted_list, predicted.flatten()), axis=-1)
            if tag_list is None:
                tag_list = tags.flatten()
            tag_list = torch.cat((tag_list, tags.flatten()), axis=-1)
            # redicted_list[0].extend(predicted.flatten())
            # tag_list[0].extend(tags.flatten())
            loss_list.append(float(loss))
        predicted_list = predicted_list.unsqueeze(0)
        tag_list = tag_list.unsqueeze(0)
        f1, p, r = ner_util.get_f1(predicted_list, tag_list, self.tag_map)
        self.log("epoch_f1", f1)
        self.log("epoch_r", r)
        self.log("epoch_p", p)
        self.log("epoch_loss", np.mean(loss_list))
        print("epoch loss %f epoch_f1 %f precision %f recall %f" %(np.mean(loss_list),f1,p,r))

    def configure_optimizers(self):
        if self.hparams.layer_lr:
            lstm_param_optimizer = list(self.lstm.named_parameters())
            crf_param_optimizer = list(self.crf.named_parameters())
            linear_param_optimizer = list(self.linear.named_parameters())
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {'params': [p for n, p in lstm_param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.hparams.weight_decay, 'lr': self.learning_rate},
                {'params': [p for n, p in lstm_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': self.learning_rate},

                {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.hparams.weight_decay, 'lr': self.crf_learning_rate},
                {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': self.crf_learning_rate},

                {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay':  self.hparams.weight_decay, 'lr': self.crf_learning_rate},
                {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': self.learning_rate}
            ]
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
                eps=self.hparams.adam_eps,
                # weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
                eps=self.hparams.adam_eps,
                weight_decay=self.hparams.weight_decay,
            )
        return optimizer
