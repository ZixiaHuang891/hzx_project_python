import torch
from torch import nn
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    AdamW,
    get_linear_schedule_with_warmup,
)

model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)


class GenreClassifierHybrid(nn.Module):
    def __init__(
        self,
        n_genres=20,
        n_numerical_features=4,
        hidden_dimension_1=728,
        hidden_dimension_2=128,
        dropout_rate=0.1,
    ):
        super(GenreClassifierHybrid, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.n_numerical_features = n_numerical_features
        self.linear_hidden_dimension_1 = hidden_dimension_1
        self.linear_hidden_dimension_2 = hidden_dimension_2
        self.dropout_rate = dropout_rate
        self.total_input_dimension = (
            self.bert.config.hidden_size + n_numerical_features
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(
                self.total_input_dimension, self.linear_hidden_dimension_1
            ),
            nn.Dropout(p=self.dropout_rate),
            nn.ReLU(),
            nn.Linear(
                self.linear_hidden_dimension_1, self.linear_hidden_dimension_2
            ),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.linear_hidden_dimension_2, n_genres),
        )

    def model_summary(self):
        summary = []
        total_params = 0

        for name, layer in self.named_modules():
            if name == "":
                continue

            layer_params = sum(
                p.numel() for p in layer.parameters() if p.requires_grad
            )
            total_params += layer_params

            layer_info = {
                "layer": name,
                "type": layer.__class__.__name__,
                "output_shape": None,
                "param_count": layer_params,
            }

            summary.append(layer_info)

        print(f"Total Parameters: {total_params:.3e}")

    def forward(self, input_ids, input_numerical, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[
            :, 0
        ]  # Get [CLS] token representation
        # Concatenate the pooled output with input numerical features along the feature dimension.
        combined = torch.cat([pooled_output, input_numerical], dim=1)
        logits = self.classifier(combined)
        return logits
