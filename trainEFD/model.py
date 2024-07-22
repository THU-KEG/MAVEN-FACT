import torch
import torch.nn as nn
from transformers import AutoModel

class RawBert(nn.Module):
    def __init__(self, model_name, tokenizer_size, num_labels=5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.resize_token_embeddings(tokenizer_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_labels)
        )
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )
        conved = outputs[0]
        logits = self.classifier(conved[:, 0, :])
        reshaped_logits = logits.view(-1, self.num_labels)
        return reshaped_logits 

class DMBert(nn.Module):
    def __init__(self, model_name, max_length, dropout, tokenizer_size, add_argument, add_relation, num_labels=5, pooling_type="cls"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.resize_token_embeddings(tokenizer_size)
        self.dropout = nn.Dropout(dropout)
        self.maxpooling = nn.MaxPool1d(max_length)
        self.add_argument = add_argument
        self.add_relation = add_relation
        self.pooling_type = pooling_type
        linear_size = self.bert.config.hidden_size * 2
        if self.add_argument:
            linear_size += self.bert.config.hidden_size
        if self.add_relation:
            linear_size += self.bert.config.hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(linear_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_labels)
        )
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, arg_ids=None, arg_mask=None, cause_ids=None, precondition_ids=None, cause_mask=None, precondition_mask=None, maskL=None, maskR=None):
        batch_size = input_ids.size(0)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )
        if self.add_argument:
            arg_outputs = self.bert(
                arg_ids,
                attention_mask=arg_mask,
            )
            arg_conved = arg_outputs[0]
            if self.pooling_type == "mean":
                pooled_arg = torch.mean(arg_conved, 1)
            elif self.pooling_type == "cls":
                pooled_arg = arg_conved[:, 0, :]
            else:
                arg_conved = arg_conved.transpose(1, 2)
                pooled_arg = self.maxpooling(arg_conved).contiguous().view(batch_size, self.bert.config.hidden_size)
        
        if self.add_relation:
            cause_outputs = self.bert(
                cause_ids,
                attention_mask=cause_mask,
            )
            cause_conved = cause_outputs[0]
            precondition_outputs = self.bert(
                precondition_ids,
                attention_mask=precondition_mask,
            )
            precondition_conved = precondition_outputs[0]
            if self.pooling_type == "mean":
                pooled_cause = torch.mean(cause_conved, 1)
                pooled_precondition = torch.mean(precondition_conved, 1)
            elif self.pooling_type == "cls":
                pooled_cause = cause_conved[:, 0, :]
                pooled_precondition = precondition_conved[:, 0, :]
            else:
                cause_conved = cause_conved.transpose(1, 2)
                pooled_cause = self.maxpooling(cause_conved).contiguous().view(batch_size, self.bert.config.hidden_size)
                precondition_conved = precondition_conved.transpose(1, 2)
                pooled_precondition = self.maxpooling(precondition_conved).contiguous().view(batch_size, self.bert.config.hidden_size)

        conved = outputs[0]
        conved = conved.transpose(1, 2)
        conved = conved.transpose(0, 1)
        L = (conved * maskL).transpose(0, 1)
        R = (conved * maskR).transpose(0, 1)
        L = L + torch.ones_like(L)
        R = R + torch.ones_like(R)
        pooledL = self.maxpooling(L).contiguous().view(batch_size, self.bert.config.hidden_size)
        pooledR = self.maxpooling(R).contiguous().view(batch_size, self.bert.config.hidden_size)
        pooled = torch.cat((pooledL, pooledR), 1)
        pooled = pooled - torch.ones_like(pooled)
        if self.add_argument:
            pooled = torch.cat((pooled, pooled_arg), 1)
        if self.add_relation:
            pooled = torch.cat((pooled, pooled_cause, pooled_precondition), 1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        reshaped_logits = logits.view(-1, self.num_labels)
        return reshaped_logits