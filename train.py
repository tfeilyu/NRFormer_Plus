import torch
import numpy as np
import argparse
import time, yaml
from src import utils
from src.trainer import Trainer
from src.model.NRFormer import NRFormer
from src.model.NRFormer_Plus import PGRT2
from src.DataProcessing import RadiationDataProcessing
import os, random
from pathlib import Path

from src.utils import upload_project_files_to_wandb

import wandb, datetime
import warnings
warnings.filterwarnings('ignore')

os.environ['WANDB_API_KEY'] = '4f2d61d3e8746b3058b2e034972550d9f91bc9dd'
# os.environ['WANDB_MODE'] = 'offline'
wandb.login()

run_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: 0; use -1 for CPU')
parser.add_argument('--dataset', type=str, default='1D-data', help='model dataset')
parser.add_argument('--model_name', type=str, default='NRFormer_Plus', help='NRFormer, NRFormer_Plus')
parser.add_argument('--batch_size', type=int, default=8, help='NRFormer, NRFormer_Plus')

parser.add_argument('--use_RevIN', type=bool, default=True, help='NRFormer, NRFormer_Plus')

parser.add_argument('--IsLocationEncoder', type=bool, default=True, help='NRFormer, NRFormer_Plus')
parser.add_argument('--IsLocationInfo', type=bool, default=True, help='NRFormer, NRFormer_Plus')

parser.add_argument('--Is_wind_angle', type=bool, default=True, help='NRFormer, NRFormer_Plus')
parser.add_argument('--Is_wind_speed', type=bool, default=True, help='NRFormer, NRFormer_Plus')
parser.add_argument('--Is_air_temperature', type=bool, default=True, help='NRFormer, NRFormer_Plus')
parser.add_argument('--Is_dew_point', type=bool, default=True, help='NRFormer, NRFormer_Plus')

parser.add_argument('--IsDayOfYearEmbedding', type=bool, default=False, help='NRFormer, NRFormer_Plus')

parser.add_argument('--num_temporal_att_layer', type=int, default=2, help='NRFormer, NRFormer_Plus')
parser.add_argument('--num_spatial_att_layer', type=int, default=2, help='NRFormer, NRFormer_Plus')

parser.add_argument('--hidden_channels', type=int, default=32, help='hidden dimension')
parser.add_argument('--end_channels', type=int, default=512, help='output projection dimension')
parser.add_argument('--temporal_dropout', type=float, default=0.1, help='dropout in temporal self-attention')
parser.add_argument('--ffn_ratio', type=int, default=4, help='FFN expansion ratio in temporal attention')
parser.add_argument('--spatial_heads', type=int, default=4, help='number of attention heads in spatial module')

parser.add_argument('--use_log_space', type=bool, default=False, help='Iter1: log-space modeling')
parser.add_argument('--use_residual', type=bool, default=False, help='Iter1: residual prediction')
parser.add_argument('--use_rain_gate', type=bool, default=False, help='Iter2: rain-aware gating')
parser.add_argument('--scheduler', type=str, default='multistep', help='Iter2: multistep or cosine')
parser.add_argument('--warmup_epochs', type=int, default=5, help='Iter2: warmup epochs for cosine scheduler')
parser.add_argument('--weight_lr', type=float, default=0.001, help='learning rate (override yaml)')

parser.add_argument('--epochs', type=int, default=300, help='number of epochs to search')
parser.add_argument('--run_times', type=int, default=1, help='number of run')
parser.add_argument('--model_des', type=str, default='111', help='save model param')
args = parser.parse_args()

with open(f'./model_settings/{args.model_name}.yaml', 'r') as config_filepath:
    hyperparams = yaml.load(config_filepath, Loader=yaml.FullLoader)
model_args = hyperparams['model']
data_args = hyperparams['data']
trainer_args = hyperparams['trainer']
if torch.cuda.is_available():
    trainer_args['gpu_count'] = torch.cuda.device_count()
    trainer_args['gpu_info'] = torch.cuda.get_device_name()
else:
    trainer_args['gpu_count'] = 0
    trainer_args['gpu_info'] = None

device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
model_args['device'] = device

# CLI args override yaml (cli_overrides applied last)
cli_overrides = {k: v for k, v in vars(args).items() if v is not None}
all_args = {**vars(args), **model_args, **data_args, **trainer_args, **cli_overrides}

# data processing
TFdata = RadiationDataProcessing(all_args)
scaler = TFdata.scaler
dataloader = TFdata.dataloader
mask_support_adj = [torch.tensor(i).to(device) for i in TFdata.adj_mx_01]

sensors_location = torch.tensor(dataloader['loc_feature']).to(device)

random.seed(all_args['seed'])
np.random.seed(all_args['seed'])
torch.manual_seed(all_args['seed'])
torch.cuda.manual_seed(all_args['seed'])
torch.cuda.manual_seed_all(all_args['seed'])

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

def main():
    if args.model_name == 'NRFormer':
        model = NRFormer(all_args, mask_support_adj)
    elif args.model_name == 'NRFormer_Plus':
        model = PGRT2(all_args, mask_support_adj)

    num_Params = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', num_Params)

    # Multi-GPU: disabled for now (buffer compatibility issues with DataParallel)
    # if torch.cuda.device_count() > 1:
    #     print(f'Using {torch.cuda.device_count()} GPUs with DataParallel')
    #     model = torch.nn.DataParallel(model)

    wandb.log({'num_params': num_Params})

    # Unique run ID: model_des + timestamp
    run_id = f'{args.model_des}_{run_time}'
    save_folder = Path('logs') / args.model_name / args.dataset / run_id
    save_folder.mkdir(parents=True, exist_ok=True)
    print(f'Experiment logs will be saved to: {save_folder}')

    train_model_path = os.path.join(save_folder, 'best_model.pt')

    engine = Trainer(model, all_args, scaler, device)

    his_valid_time = []
    his_train_time = []
    his_valid_loss = []
    # Epoch history for analysis
    epoch_history = {'epoch': [], 'train_loss': [], 'train_mae': [], 'train_rmse': [],
                     'valid_loss': [], 'valid_mae': [], 'valid_rmse': [],
                     'lr': [], 'grad_norm': []}
    min_valid_loss = 1000
    best_epoch = 0
    all_start_time = time.time()

    # Save config snapshot with all hyperparameters
    import json
    config_path = save_folder / 'config.json'
    save_config = {k: str(v) if isinstance(v, (torch.device, type)) else v for k, v in all_args.items()}
    save_config['num_params'] = num_Params
    save_config['run_id'] = run_id
    with open(config_path, 'w') as f:
        json.dump(save_config, f, indent=2, default=str)

    print("start training...\n", flush=True)
    for epoch in range(best_epoch+1, best_epoch+args.epochs+1):
        epoch_start_time = time.time()

        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        train_grad_norm = []
        train_start_time = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            train_x = torch.Tensor(x).to(device)
            train_x = train_x.transpose(1, 3)
            train_y = torch.Tensor(y).to(device)
            train_y = train_y.transpose(1, 3)
            train_metrics = engine.train_weight(train_x, sensors_location, train_y[:, 0, :, :])
            train_loss.append(train_metrics[0])
            train_mae.append(train_metrics[1])
            train_mape.append(train_metrics[2])
            train_rmse.append(train_metrics[3])
            train_grad_norm.append(train_metrics[4])
        engine.weight_scheduler.step()

        train_end_time = time.time()
        t_time = train_end_time - train_start_time
        his_train_time.append(t_time)

        valid_loss = []
        valid_mae = []
        valid_mape = []
        valid_rmse = []
        valid_start_time = time.time()
        for iter, (x, y) in enumerate(dataloader['valid_loader'].get_iterator()):
            val_x = torch.Tensor(x).to(device)
            val_x = val_x.transpose(1, 3)
            val_y = torch.Tensor(y).to(device)
            val_y = val_y.transpose(1, 3)
            val_metrics = engine.eval(val_x, sensors_location, val_y[:, 0, :, :])
            valid_loss.append(val_metrics[0])
            valid_mae.append(val_metrics[1])
            valid_mape.append(val_metrics[2])
            valid_rmse.append(val_metrics[3])
        # engine.arch_scheduler.step()

        valid_end_time = time.time()
        v_time = valid_end_time - valid_start_time
        his_valid_time.append(v_time)

        epoch_time = time.time() - epoch_start_time

        mean_train_loss = np.mean(train_loss)
        mean_train_mae = np.mean(train_mae)
        mean_train_mape = np.mean(train_mape)
        mean_train_rmse = np.mean(train_rmse)

        mean_valid_loss = np.mean(valid_loss)
        mean_valid_mae = np.mean(valid_mae)
        mean_valid_mape = np.mean(valid_mape)
        mean_valid_rmse = np.mean(valid_rmse)

        his_valid_loss.append(mean_valid_loss)

        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(now_time)

        print('{}, Epoch: {:03d}, Epoch Training Time: {:.4f}'.format(args.dataset, epoch, epoch_time))
        wandb.log({'Epoch Time': epoch_time})

        log_loss = 'Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Train Time: {:.4f}\n' \
                   'Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Valid Time: {:.4f}\n'
        print(log_loss.format(mean_train_loss, mean_train_mae, mean_train_mape, mean_train_rmse, t_time,
                              mean_valid_loss, mean_valid_mae, mean_valid_mape, mean_valid_rmse, v_time),flush=True)

        current_lr = engine.get_lr()
        mean_grad_norm = np.mean(train_grad_norm)

        # Record epoch history
        epoch_history['epoch'].append(epoch)
        epoch_history['train_loss'].append(mean_train_loss)
        epoch_history['train_mae'].append(mean_train_mae)
        epoch_history['train_rmse'].append(mean_train_rmse)
        epoch_history['valid_loss'].append(mean_valid_loss)
        epoch_history['valid_mae'].append(mean_valid_mae)
        epoch_history['valid_rmse'].append(mean_valid_rmse)
        epoch_history['lr'].append(current_lr)
        epoch_history['grad_norm'].append(mean_grad_norm)

        log_dict = {'Epoch': epoch,
                   'Epoch Search/Train Loss': mean_train_loss, 'Epoch Search/Train MAE': mean_train_mae,
                   'Epoch Search/Train MAPE': mean_train_mape, 'Epoch Search/Train RMSE': mean_train_rmse,
                   'Epoch Search/Train Time': t_time,
                   'Epoch Valid Loss': mean_valid_loss, 'Epoch Valid MAE': mean_valid_mae,
                   'Epoch Valid MAPE': mean_valid_mape, 'Epoch Valid RMSE': mean_valid_rmse,
                   'Epoch Valid Time': v_time,
                   'learning_rate': current_lr, 'grad_norm': mean_grad_norm}

        # Physics diagnostics (every 10 epochs)
        if epoch % 10 == 0:
            physics_diag = engine.get_physics_diagnostics()
            if physics_diag:
                log_dict.update({f'physics/{k}': v for k, v in physics_diag.items()})

        wandb.log(log_dict)

        if mean_valid_loss < min_valid_loss:
            best_epoch = epoch
            engine.save(train_model_path, best_epoch)
            print('[eval]\tepoch {}\tsave parameters to {}\n'.format(best_epoch, train_model_path))
            min_valid_loss = mean_valid_loss

        elif all_args['early_stop'] and epoch - best_epoch > all_args['early_stop_steps']:
            print('-' * 40)
            print('Early Stopped, best train epoch:', best_epoch)
            print('-' * 40)
            break

    all_end_time = time.time()

    # Save epoch history CSV
    import pandas as pd
    history_df = pd.DataFrame(epoch_history)
    history_path = save_folder / 'epoch_history.csv'
    history_df.to_csv(history_path, index=False)
    print(f"Epoch history saved to {history_path}")

    # Generate training curves
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(history_df['epoch'], history_df['train_loss'], label='Train')
        axes[0, 0].plot(history_df['epoch'], history_df['valid_loss'], label='Valid')
        axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend(); axes[0, 0].set_title('Loss')

        axes[0, 1].plot(history_df['epoch'], history_df['train_mae'], label='Train')
        axes[0, 1].plot(history_df['epoch'], history_df['valid_mae'], label='Valid')
        axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend(); axes[0, 1].set_title('MAE')

        axes[1, 0].plot(history_df['epoch'], history_df['lr'])
        axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('LR')
        axes[1, 0].set_title('Learning Rate')

        axes[1, 1].plot(history_df['epoch'], history_df['grad_norm'])
        axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Grad Norm')
        axes[1, 1].set_title('Gradient Norm')

        plt.tight_layout()
        fig_path = save_folder / 'training_curves.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        wandb.log({'training_curves': wandb.Image(str(fig_path))})
        print(f"Training curves saved to {fig_path}")
    except Exception as e:
        print(f"Warning: could not generate plots: {e}")

    print('\n')
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    best_id = np.argmin(his_valid_loss)
    print("Training finished.")
    best_epoch = engine.load(train_model_path)
    print("The valid loss on best trained model is {}, epoch:{}\n"
          .format(str(round(his_valid_loss[best_id], 4)), best_epoch))

    print("All Training Time: {:.4f} secs".format(all_end_time-all_start_time))
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(his_train_time)))
    print("Average Inference Time: {:.4f} secs/epoch".format(np.mean(his_valid_time)))
    wandb.log({'All Running Time': all_end_time-all_start_time,
               'Average Epoch Search/Train Time': np.mean(his_train_time),
               'Average Epoch Inference Time': np.mean(his_valid_time)})

    print('\n')
    print("Best Train Model Loaded")

    outputs = []
    true_valid_y = []
    for iter, (x, y) in enumerate(dataloader['valid_loader'].get_iterator()):
        valid_x = torch.Tensor(x).to(device)
        valid_x = valid_x.transpose(1, 3)
        valid_y = torch.Tensor(y).to(device)
        valid_y = valid_y.transpose(1, 3)[:, 0, :, :]

        with torch.no_grad():
            # input (64, 2, 207, 12)
            preds = engine.model(valid_x, sensors_location)
            # output (64, 1, 207, 12)

        outputs.append(preds.squeeze(dim=1))
        true_valid_y.append(valid_y)
    valid_yhat = torch.cat(outputs, dim=0)
    true_valid_y = torch.cat(true_valid_y, dim=0)
    # valid_yhat = valid_yhat[:valid_y.size(0), ...]
    valid_pred = scaler.inverse_transform(valid_yhat)
    if all_args.get('use_log_space', False):
        valid_pred = torch.expm1(valid_pred)
    valid_pred = torch.clamp(valid_pred, min=0., max=scaler.max_value)
    valid_mae, valid_mape, valid_rmse = utils.metric(valid_pred, true_valid_y)

    log = '{} Average Performance on Valid Data - Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}'
    print(log.format(all_args['out_length'], valid_mae, valid_mape, valid_rmse))
    wandb.log({"valid MAE": valid_mae, "valid MAPE": valid_mape, "valid RMSE": valid_rmse})

    outputs = []
    true_test_y = []
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        test_x = torch.Tensor(x).to(device)
        test_x = test_x.transpose(1, 3)
        test_y = torch.Tensor(y).to(device)
        test_y = test_y.transpose(1, 3)[:, 0, :, :]

        with torch.no_grad():
            # input (64, 2, 207, 12)
            preds = engine.model(test_x, sensors_location)
            # output (64, 1, 207, 12)

        outputs.append(preds.squeeze(dim=1))
        true_test_y.append(test_y)
    test_yhat = torch.cat(outputs, dim=0)
    true_test_y = torch.cat(true_test_y, dim=0)
    # test_yhat = test_yhat[:test_y.size(0), ...]
    test_pred = scaler.inverse_transform(test_yhat)
    if all_args.get('use_log_space', False):
        test_pred = torch.expm1(test_pred)
    test_pred = torch.clamp(test_pred, min=0., max=scaler.max_value)
    test_mae, test_mape, test_rmse = utils.metric(test_pred, true_test_y)

    log = '{} Average Performance on Test Data - Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f} \n'
    print(log.format(all_args['out_length'], test_mae, test_mape, test_rmse))
    wandb.log({"test MAE": test_mae, "test MAPE": test_mape, "test RMSE": test_rmse})

    print('Single steps test:')
    mae = []
    mape = []
    rmse = []
    for i in step_list:
        i=i-1
        pred_singlestep = test_pred[:, :, i]
        real = true_test_y[:, :, i]
        metrics_single = utils.metric(pred_singlestep, real)
        log = 'horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics_single[0], metrics_single[1], metrics_single[2]))
        wandb.log({str(i + 1) + '_MAE_single': metrics_single[0],
                   str(i + 1) + '_MAPE_single': metrics_single[1],
                   str(i + 1) + '_RMSE_single': metrics_single[2]})
        mae.append(metrics_single[0])
        mape.append(metrics_single[1])
        rmse.append(metrics_single[2])

    print('\nAverage steps test:')
    mae_avg = []
    mape_avg = []
    rmse_avg = []
    for i in step_list:
        pred_avg_step = test_pred[:, :, :i]
        real = true_test_y[:, :, :i]
        metrics_avg = utils.metric(pred_avg_step, real)
        log = 'average {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i, metrics_avg[0], metrics_avg[1], metrics_avg[2]))
        wandb.log({str(i) + '_MAE_avg': metrics_avg[0],
                   str(i) + '_MAPE_avg': metrics_avg[1],
                   str(i) + '_RMSE_avg': metrics_avg[2]})
        mae_avg.append(metrics_avg[0])
        mape_avg.append(metrics_avg[1])
        rmse_avg.append(metrics_avg[2])

    # Save test results summary
    results = {
        'best_epoch': int(best_epoch),
        'num_params': num_Params,
        'valid': {'MAE': valid_mae, 'MAPE': valid_mape, 'RMSE': valid_rmse},
        'test': {'MAE': test_mae, 'MAPE': test_mape, 'RMSE': test_rmse},
        'per_horizon': {}
    }
    for idx, step in enumerate(step_list):
        results['per_horizon'][f'step_{step}'] = {
            'MAE': mae[idx], 'MAPE': mape[idx], 'RMSE': rmse[idx],
            'MAE_avg': mae_avg[idx], 'MAPE_avg': mape_avg[idx], 'RMSE_avg': rmse_avg[idx]
        }
    results_path = save_folder / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {results_path}')

    wandb.finish()
    return valid_mae, valid_mape, valid_rmse, test_mae, test_mape, test_rmse, mae, mape, rmse, mae_avg, mape_avg, rmse_avg


if __name__ == "__main__":

    valid_MAE = []
    valid_MAPE = []
    valid_RMSE = []

    test_MAE = []
    teat_MAPE = []
    test_RMSE = []

    MAE = []
    MAPE = []
    RMSE = []

    MAE_avg = []
    MAPE_avg = []
    RMSE_avg = []

    if all_args['out_length'] == 12:
        step_list = [3,6,9,12]
    elif all_args['out_length'] == 24:
        step_list = [6,9,12,24]

    for run_num in range(args.run_times):

        run_name = '{}_train_{}_{}'.format(args.model_name, all_args['model_des'], run_time)
        wandb.init(
            # set the wandb project where this run will be logged
            project='NRFormer_Plus_TKDE_26',
            name=run_name,
            config=all_args
        )
        upload_project_files_to_wandb()

        # track hyperparameters and run metadata
        vm1, vm2, vm3, tm1, tm2, tm3, m1, m2, m3, ma1, ma2, ma3, = main()

        valid_MAE.append(vm1)
        valid_MAPE.append(vm2)
        valid_RMSE.append(vm3)

        test_MAE.append(tm1)
        teat_MAPE.append(tm2)
        test_RMSE.append(tm3)

        MAE.append(m1)
        MAPE.append(m2)
        RMSE.append(m3)

        MAE_avg.append(ma1)
        MAPE_avg.append(ma2)
        RMSE_avg.append(ma3)

    mae_single = np.mean(np.array(MAE), 0)
    mape_single = np.mean(np.array(MAPE), 0)
    rmse_single = np.mean(np.array(RMSE), 0)

    mae_single_std = np.std(np.array(MAE), 0)
    mape_single_std = np.std(np.array(MAPE), 0)
    rmse_single_std = np.std(np.array(RMSE), 0)

    mae_avg = np.mean(np.array(MAE_avg), 0)
    mape_avg = np.mean(np.array(MAPE_avg), 0)
    rmse_avg = np.mean(np.array(RMSE_avg), 0)

    mae_avg_std = np.std(np.array(MAE_avg), 0)
    mape_avg_std = np.std(np.array(MAPE_avg), 0)
    rmse_avg_std = np.std(np.array(RMSE_avg), 0)

    print('\n')
    print(args.dataset)
    print('\n')

    print('valid\t MAE\t RMSE\t MAPE')
    log = 'mean:\t {:.4f}\t {:.4f}\t {:.4f}'
    print(log.format(np.mean(valid_MAE), np.mean(valid_RMSE), np.mean(valid_MAPE)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(valid_MAE), np.std(valid_RMSE), np.std(valid_MAPE)))
    print('\n')

    print('test\t MAE\t RMSE\t MAPE')
    log = 'mean:\t {:.4f}\t {:.4f}\t {:.4f}'
    print(log.format(np.mean(test_MAE), np.mean(test_RMSE), np.mean(teat_MAPE)))
    log = 'std:\t {:.4f}\t {:.4f}\t {:.4f}'
    print(log.format(np.std(test_MAE), np.std(test_RMSE), np.std(teat_MAPE)))
    print('\n')

    print('single test:')
    print('horizon\t MAE-mean\t RMSE-mean\t MAPE-mean\t MAE-std\t RMSE-std\t MAPE-std')
    for i in range(4):
        log = '{:d}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}'
        print(log.format(step_list[i], mae_single[i], rmse_single[i], mape_single[i], mae_single_std[i],
                         rmse_single_std[i], mape_single_std[i]))

    print('\n')
    print('avg test:')
    print('average\t MAE-mean\t RMSE-mean\t MAPE-mean\t MAE-std\t RMSE-std\t MAPE-std')
    for i in range(4):
        log = '{:d}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}'
        print(log.format(step_list[i], mae_avg[i], rmse_avg[i], mape_avg[i], mae_avg_std[i],
                         rmse_avg_std[i], mape_avg_std[i]))

    print('Train Finish!\n')
