"""
Evaluation
"""
import os
import math
import tempfile
import torch
import pickle
import logging

# Third Party
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
import numpy as np
import pandas as pd
from torch.utils.data import Subset, ConcatDataset

# First Party
from tsfm_public.models.tinytimemixer.utils import (
    count_parameters,
    plot_preds,
)

# Local
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.callbacks import TrackingCallback
from tsfm_public.toolkit.time_series_forecasting_pipeline import TimeSeriesPipeline
import warnings
from tsfm_public.models.tinytimemixer.utils.ttm_utils import get_data

# Suppress all warnings
warnings.filterwarnings("ignore")

SEED = 42
set_seed(SEED)
TTM_MODEL_REVISION = "main"  # versions of pre-trained TTM model
DATA_ROOT_PATH = './dataset/'
OUT_DIR = 'tsfm_save/'

def cal_cvrmse(pred, true):
    pred = np.array(pred)
    true = np.array(true)
    return np.power(np.square(pred - true).sum() / pred.shape[0], 0.5) / (true.sum() / pred.shape[0])

def cal_mae(pred, true):
    pred = np.array(pred)
    true = np.array(true)
    return np.mean(np.abs(pred - true))

def denormalize(load, load_max, load_min):
    return load * (load_max - load_min) + load_min

def get_pretrained_model(ckpt_path='ibm/TTM', freeze_backbone=True):
    if prediction_filter_length is None:
        pretrained_model = TinyTimeMixerForPrediction.from_pretrained(
            ckpt_path, revision=TTM_MODEL_REVISION, config="config.json",
        )
    elif prediction_filter_length <= forecast_length:
        pretrained_model = TinyTimeMixerForPrediction.from_pretrained(
            ckpt_path, revision=TTM_MODEL_REVISION, prediction_filter_length=prediction_filter_length,
            config="config.json",
        )
    else:
        raise ValueError(f"`prediction_filter_length` should be <= `forecast_length")

    if freeze_backbone:
        print(
            "Number of params before freezing backbone",
            count_parameters(pretrained_model),
        )

        # Freeze the backbone of the model
        for param in pretrained_model.backbone.parameters():
            param.requires_grad = False

        # Count params
        print(
            "Number of params after freezing the backbone",
            count_parameters(pretrained_model),
        )
    return pretrained_model

def model_finetuning(model, dataset_name, batch_size, dset_train, dset_val,
                     fewshot_percent, learning_rate, num_epochs, save_dir=OUT_DIR):

    out_dir = os.path.join(save_dir, dataset_name)

    print("-" * 20, f"Running few-shot {fewshot_percent}%", "-" * 20)
    print(f"Using learning rate = {learning_rate}")
    finetune_forecast_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "output"),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=8,
        report_to=None,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(out_dir, "logs"),  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
        use_cpu=False,
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
    )
    tracking_callback = TrackingCallback()

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch=math.ceil(len(dset_train) / (batch_size)),
    )

    finetune_forecast_trainer = Trainer(
        model=model,
        args=finetune_forecast_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
        callbacks=[early_stopping_callback, tracking_callback],
        optimizers=(optimizer, scheduler),
    )

    # Fine tune
    finetune_forecast_trainer.train()
    return finetune_forecast_trainer

def eval_model(model, dset_test, prediction_filter_length, tsp):
    trainer = Trainer(model=model)

    output = trainer.predict(dset_test)
    output = output[0][0]

    true_list, pred_list = [], []
    for i in range(0, len(dset_test), prediction_filter_length):
        true_list.append(np.array(dset_test[i]["future_values"][:prediction_filter_length, :]))
        pred_list.append(np.array(output[i, :, :]))
    true, pred = np.array(true_list).flatten(), np.array(pred_list).flatten()

    # inverse scale
    df_true = pd.DataFrame(true, columns=['power'])
    df_pred = pd.DataFrame(pred, columns=['power'])
    df_inv_true = tsp.inverse_scale_targets(df_true)
    df_inv_pred = tsp.inverse_scale_targets(df_pred)
    true = np.array(df_inv_true)
    pred = np.array(df_inv_pred)

    cv_rmse, mae = cal_cvrmse(pred, true), cal_mae(pred, true)
    print("CVRMSE: {:.4f}, \t MAE: {:.4f}".format(cv_rmse, mae))

    # vis
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(true, color="red", label="ture")
    plt.plot(pred, color="blue", label="prediction")
    plt.legend()
    plt.show()

    return cv_rmse, mae


if __name__ == '__main__':
    # note: parameters that need to be adjusted
    dataset_name = "Genome"
    evaluation_mode = 'zeroshot'  # zeroshot, fewshot
    ckpt_path = 'tsfm_save/clft/output/checkpoint-v0.8'
    file_path = './dataset/Genome/Fox_office_Joy.csv'

    forecast_length = 96
    prediction_filter_length = 96
    context_length = 512
    batch_size = 16
    fewshot_percent = 100
    n_steps = 5
    learning_rate = 1e-3

    # note: evaluation
    tsp, dset_train, dset_val, dset_test = get_data(dataset_name=dataset_name,
                                                    file_name=file_path,
                                                    context_length=context_length,
                                                    forecast_length=forecast_length,
                                                    fewshot_fraction=fewshot_percent / 100,
                                                    data_root_path=DATA_ROOT_PATH)

    # get pretrained model
    model = get_pretrained_model(ckpt_path=ckpt_path, freeze_backbone=True)

    # if fewshot, fine-tune the model with target building data
    if evaluation_mode == 'fewshot':
        trainer = model_finetuning(model=model,
                                   dataset_name=dataset_name,
                                   batch_size=batch_size,
                                   dset_train=dset_train,
                                   dset_val=dset_val,
                                   fewshot_percent=fewshot_percent,
                                   learning_rate=learning_rate,
                                   num_epochs=n_steps,
                                   save_dir=OUT_DIR)
        model = trainer.model

    # eval model
    cv_rmse, mae = eval_model(model, dset_test, prediction_filter_length, tsp)
