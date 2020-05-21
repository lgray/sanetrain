main_loop = '''import sanetrain
import mlflow
import mlflow.pytorch
import os.path

import torch
import torch.nn
import torch.nn.functional as F

device = sanetrain.device.device

{define_datasets}

{data_unpacker}

with mlflow.start_run() as run:
    mlflow.log_artifact(os.path.realpath(__file__), "script")

    train_loader = DataLoader(train_dataset, {train_loader_params})
    test_loader = DataLoader(test_dataset, {test_loader_params})
    
    train_step = {train_step}
    
    train_epoch = {train_epoch}
    
    test_step = {test_step}
    
    test_epoch = {test_epoch}
    
    display_stats = {display_stats}
    
    {model_import}
    model = {model_name}({model_params}).to(device)
    
    optimizer = {optimizer}(model.parameters(), {optimizer_params})
    
    lossf = {loss_function}
    
    for epoch in range(1, {epochs} + 1):
        train_epoch(optimizer, model, lossf, train_loader, data_unpacker,
                    train_step, epoch=epoch, device=device,
                    {train_kwargs})
    
        test_stats = test_epoch(model, lossf, test_loader, data_unpacker,
                                test_step, epoch=epoch, device=device,
                                {test_kwargs})
    
        mlflow.log_metric("loss", test_stats["loss"], step=epoch)
        mlflow.log_metric("test_acc", test_stats["correct"]/test_stats["total"], step=epoch)

        display_stats(**test_stats)
    mlflow.pytorch.log_model(model, "models")
'''
