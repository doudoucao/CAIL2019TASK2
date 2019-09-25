from pytorch_pretrained_bert.tokenization import BertTokenizer, WordpieceTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertPreTrainedModel, BertModel, BertConfig, BertForMaskedLM, BertForSequenceClassification
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from capsule_network import Caps_Layer
import torch.nn as nn
import torch
import os
import sys
from loss import binary_cross_entropy, FocalLoss


module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=20):
        super(BertForMultiLabelClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.num_capsule = 10
        self.dim_capsule = 16
        self.caps = Caps_Layer(batch_size=12, input_dim_capsule=config.hidden_size, num_capsule=10, dim_capsule=16,
                               routings=5)
        self.dense = nn.Linear(self.num_capsule*self.dim_capsule, num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        last_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # last_output = torch.cuda.FloatTensor(last_output)
        # attention_mask = torch.cuda.FloatTensor(attention_mask)
        pooled_output = torch.sum(last_output * attention_mask.float().unsqueeze(2), dim=1) / torch.sum(attention_mask.float(), dim=1, keepdim=True)
        '''
        batch_size = input_ids.size(0)
        caps_output = self.caps(last_output)  # (batch_size, num_capsule, dim_capsule)
        caps_output = caps_output.view(batch_size, -1)  # (batch_size, num_capsule*dim_capsule)
        caps_dropout = self.dropout(caps_output)
        logits = self.dense(caps_dropout)
        '''

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            # loss_fct = BCEWithLogitsLoss()
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            alpha = 0.75
            gamma = 3

            # focal loss

            x = logits.view(-1, self.num_labels)
            t = labels.view(-1, self.num_labels)
            '''
            p = x.sigmoid()
            pt = p*t + (1-p)*(1-t)
            w = alpha*t + (1-alpha)*(1-t)
            w = w*(1-pt).pow(gamma)
            # return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)
            return binary_cross_entropy(x, t, weight=w, smooth_eps=0.1, from_logits=True)
            '''
            loss_fct = FocalLoss(logits=True)
            loss = loss_fct(x, t)
            return loss
        else:
            return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True
