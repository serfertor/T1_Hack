import torch
from torch import nn
from transformers import BertModel, BertTokenizer

class BERTClassifier(nn.Module):
    """
    Архитектура модели, идентичная той, что использовалась при обучении.
    """
    def __init__(self, n_classes):
        super(BERTClassifier, self).__init__()
        # Загружаем предобученную модель BERT
        self.bert = BertModel.from_pretrained('DeepPavlov/rubert-base-cased', return_dict=False)
        # Слой Dropout для регуляризации
        self.drop = nn.Dropout(p=0.3)
        # Полносвязный слой для классификации
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        # Пропускаем данные через BERT
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Применяем Dropout и классификационный слой
        output = self.drop(pooled_output)
        return self.out(output)