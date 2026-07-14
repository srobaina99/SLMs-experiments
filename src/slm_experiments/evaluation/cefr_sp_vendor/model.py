"""Vendored CEFR-SP level estimators (trimmed for inference)."""

from __future__ import annotations

import torch
from torch import nn

from .model_base import LevelEstimaterBase
from .util import convert_numeral_to_six_levels, mean_pooling


class LevelEstimaterClassification(LevelEstimaterBase):
    def __init__(
        self,
        corpus_path,
        test_corpus_path,
        pretrained_model,
        problem_type,
        with_ib,
        with_loss_weight,
        attach_wlv,
        num_labels,
        word_num_labels,
        alpha,
        ib_beta,
        batch_size,
        learning_rate,
        warmup,
        lm_layer,
    ):
        super().__init__(
            corpus_path,
            test_corpus_path,
            pretrained_model,
            with_ib,
            attach_wlv,
            num_labels,
            word_num_labels,
            alpha,
            batch_size,
            learning_rate,
            warmup,
            lm_layer,
        )
        self.save_hyperparameters()

        self.problem_type = problem_type
        self.with_loss_weight = with_loss_weight
        self.ib_beta = ib_beta
        self.dropout = nn.Dropout(0.1)

        if self.problem_type == "regression":
            self.slv_classifier = nn.Linear(self.lm.config.hidden_size, 1)
            self.loss_fct = nn.MSELoss()
        else:
            self.slv_classifier = nn.Linear(self.lm.config.hidden_size, self.CEFR_lvs)
            if self.with_loss_weight:
                train_sentlv_weights = self.precompute_loss_weights()
                self.loss_fct = nn.CrossEntropyLoss(weight=train_sentlv_weights)
            else:
                self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, inputs):
        outputs, _information_loss = self.encode(inputs)
        outputs = mean_pooling(outputs, attention_mask=inputs["attention_mask"])
        logits = self.slv_classifier(self.dropout(outputs))

        if self.problem_type == "regression":
            predictions = convert_numeral_to_six_levels(
                logits.detach().clone().cpu().numpy()
            )
        else:
            predictions = torch.argmax(
                torch.softmax(logits.detach().clone(), dim=1), dim=1, keepdim=True
            )

        loss = None
        if "slabels_high" in inputs:
            if self.problem_type == "regression":
                labels = (inputs["slabels_high"] + inputs["slabels_low"]) / 2
                cls_loss = self.loss_fct(logits.squeeze(), labels.squeeze())
            else:
                labels = self.get_gold_labels(
                    predictions,
                    inputs["slabels_low"].detach().clone(),
                    inputs["slabels_high"].detach().clone(),
                )
                cls_loss = self.loss_fct(logits.view(-1, self.CEFR_lvs), labels.view(-1))

            loss = cls_loss
            logs = {"loss": cls_loss}
            predictions = predictions.cpu().numpy()
            return (loss, predictions, logs) if loss is not None else predictions

        predictions = predictions.cpu().numpy()
        return predictions

    def step(self, batch):
        loss, _predictions, logs = self.forward(batch)
        return loss, logs

    def _shared_eval_step(self, batch):
        loss, predictions, logs = self.forward(batch)
        gold_labels_low = batch["slabels_low"].cpu().detach().clone().numpy()
        gold_labels_high = batch["slabels_high"].cpu().detach().clone().numpy()
        golds_predictions = {
            "gold_labels_low": gold_labels_low,
            "gold_labels_high": gold_labels_high,
            "pred_labels": predictions,
        }
        del loss
        return logs, golds_predictions

    def training_step(self, batch, batch_idx):
        del batch_idx
        loss, logs = self.step(batch)
        self.log_dict({f"train_{k}": v for k, v in logs.items()})
        return loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        logs, golds_predictions = self._shared_eval_step(batch)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return golds_predictions

    def test_step(self, batch, batch_idx):
        del batch_idx
        logs, golds_predictions = self._shared_eval_step(batch)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})
        return golds_predictions


class LevelEstimaterContrastive(LevelEstimaterBase):
    """Official CEFR-SP contrastive prototype model (Zenodo checkpoint)."""

    def __init__(
        self,
        corpus_path,
        test_corpus_path,
        pretrained_model,
        problem_type,
        with_ib,
        with_loss_weight,
        attach_wlv,
        num_labels,
        word_num_labels,
        num_prototypes,
        alpha,
        ib_beta,
        batch_size,
        learning_rate,
        warmup,
        lm_layer,
    ):
        super().__init__(
            corpus_path,
            test_corpus_path,
            pretrained_model,
            with_ib,
            attach_wlv,
            num_labels,
            word_num_labels,
            alpha,
            batch_size,
            learning_rate,
            warmup,
            lm_layer,
        )
        self.save_hyperparameters()

        self.problem_type = problem_type
        self.num_prototypes = num_prototypes
        self.with_loss_weight = with_loss_weight
        self.ib_beta = ib_beta

        self.prototype = nn.Embedding(
            self.CEFR_lvs * self.num_prototypes, self.lm.config.hidden_size
        )

        if self.with_loss_weight:
            loss_weights = self.precompute_loss_weights()
            self.loss_fct = nn.CrossEntropyLoss(weight=loss_weights)
        else:
            self.loss_fct = nn.CrossEntropyLoss()

    def contrastive_logits(self, batch):
        """Return class logits (B, 6) without requiring gold labels."""
        outputs, _information_loss = self.encode(batch)
        outputs = mean_pooling(outputs, attention_mask=batch["attention_mask"])
        outputs = torch.nn.functional.normalize(outputs)
        positive_prototypes = torch.nn.functional.normalize(self.prototype.weight)
        logits = torch.mm(outputs, positive_prototypes.T)
        logits = logits.reshape((-1, self.num_prototypes, self.CEFR_lvs))
        return logits.mean(dim=1)

    def forward(self, batch):
        logits = self.contrastive_logits(batch)
        predictions = torch.argmax(
            torch.softmax(logits.detach().clone(), dim=1), dim=1, keepdim=True
        )

        loss = None
        if "slabels_high" in batch:
            labels = self.get_gold_labels(
                predictions,
                batch["slabels_low"].detach().clone(),
                batch["slabels_high"].detach().clone(),
            )
            cls_loss = self.loss_fct(logits.view(-1, self.CEFR_lvs), labels.view(-1))
            loss = cls_loss
            logs = {"loss": loss}
            predictions = predictions.cpu().numpy()
            return (loss, predictions, logs)

        return predictions.cpu().numpy()

    def _shared_eval_step(self, batch):
        loss, predictions, logs = self.forward(batch)
        gold_labels_low = batch["slabels_low"].cpu().detach().clone().numpy()
        gold_labels_high = batch["slabels_high"].cpu().detach().clone().numpy()
        golds_predictions = {
            "gold_labels_low": gold_labels_low,
            "gold_labels_high": gold_labels_high,
            "pred_labels": predictions,
        }
        del loss
        return logs, golds_predictions

    def training_step(self, batch, batch_idx):
        del batch_idx
        loss, _predictions, logs = self.forward(batch)
        self.log_dict({f"train_{k}": v for k, v in logs.items()})
        return loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        logs, golds_predictions = self._shared_eval_step(batch)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return golds_predictions

    def test_step(self, batch, batch_idx):
        del batch_idx
        logs, golds_predictions = self._shared_eval_step(batch)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})
        return golds_predictions
