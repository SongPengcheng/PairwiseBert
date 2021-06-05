import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BERT_INPUTS_DOCSTRING
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable

class TextCNN(nn.Module):
    def __init__(self,filter_sizes,embedding_size,filter_num,dropout_prop,label_num):
        super().__init__()
        self.filter_sizes = filter_sizes
        self.embedding_size = embedding_size
        self.filter_num = filter_num
        self.label_num = label_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.convs = [nn.Conv2d(1, filter_num, (fsize, embedding_size)).to(self.device) for fsize in filter_sizes]
        self.dropout = nn.Dropout(dropout_prop)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, self.label_num).to(self.device)  ##全连接层
    def forward(self, x):
        x = x.unsqueeze(1)  # (N,Ci,W,D)

        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]  # len(Ks)*(N,Knum,W)
        x = [torch.max_pool1d(line, line.size(2)).squeeze(2) for line in x]  # len(Ks)*(N,Knum)
        x = torch.cat(x, 1)  # (N,Knum*len(Ks))
        x = self.dropout(x)
        logit = self.fc(x)
        return logit

class BertWithTextCNN(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bert = BertModel(config)
        self.textcnn = TextCNN(
            filter_sizes=[3,4,5],
            filter_num=2,
            embedding_size=config.hidden_size,
            dropout_prop=config.hidden_dropout_prob,
            label_num=2
        )
        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        self.textcnn.to(sequence_output.device)
        logits = self.textcnn(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)



