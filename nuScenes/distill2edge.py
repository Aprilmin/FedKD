from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DistributedSampler,DataLoader
from lora_model import DriveMLLM
import deepspeed.comm as dist
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from nuscenes_dataset import Cloud2EdgesMultiFrameDataset
from local_update_for_one_edge import save_filtered_params

import torch
import argparse
import os

from utils import setup_seed



def cloud_distill2_edge(ds_config,args,prev_edge_w_path,train_input_file,check_path,communication):
    cloud_start_time=datetime.now()
    proxy = DriveMLLM(rank=args.rank, lora_alpha=args.lora_alpha, peft=args.peft.lower(),pretrained_model_path=args.edge_model_path)

    this_prev_edge_w=torch.load(prev_edge_w_path, weights_only=False)['model']
    proxy.load_state_dict(this_prev_edge_w, strict=False)

    proxy.freeze_block(proxy, freeze_personal_block=True, freeze_global_block=False,freeze_shared_block=False)
    personal_params_for_recover=proxy.set_personal_disable(proxy)

    this_prev_edge_w={param_name: None for param_name in proxy._get_all_peft_params(proxy).keys()}
    optimizer = DeepSpeedCPUAdam(list(filter(lambda p: p.requires_grad, proxy.model.parameters()))+ list(proxy.loss_fn.parameters()), lr=args.d2edge_lr,
                                 betas=(0.9, 0.999), eps=1e-8)

    proxy, optimizer, _, scheduler = deepspeed.initialize(
        model=proxy,
        optimizer=optimizer,
        model_parameters=filter(lambda p: p.requires_grad, proxy.model.parameters()),
        # training_data=train_dset,
        config_params=ds_config,
        lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
    )

    train_dest = Cloud2EdgesMultiFrameDataset(sample_root=args.data_path, input_file=train_input_file,processor=proxy.processor,client_idx=args.client_idx,check_path=check_path, communication=communication,model_name=proxy.model_name)
    sampler = DistributedSampler(train_dest, num_replicas=args.gpus, rank=torch.distributed.get_rank())
    train_dataloader = DataLoader(train_dest, shuffle=False, batch_size=1,sampler=sampler, num_workers=0, collate_fn=train_dest.custom_collate_fn)

    proxy.train()


    best_train_loss=1000000
    speed_std = 1.8046681286146409
    speed_mean = 2.5373214399490234
    curv_std = 0.07511833558384672
    curv_mean = 0.021407328091665685
    speed_min = 0
    speed_max = 10.082848494377343
    for epoch in range(args.cloud_epochs):
        sampler.set_epoch(epoch)
        epoch_start_time = datetime.now()
        epoch_loss = 0


        progress_bar = tqdm(enumerate(train_dataloader),total=len(train_dataloader))

        for step, (inputs,labels,teachers_logits_res,future_speed_and_curvatur, future_trajs) in progress_bar:
            step_start_time=datetime.now()

            inputs = {
                k: v.to(proxy.device).requires_grad_(True) if k == "pixel_values" else v.to(proxy.device)
                for k, v in inputs.items()
            }


            labels = labels.to(proxy.device)

            future_speed_and_curvatur = torch.tensor(future_speed_and_curvatur, device=proxy.device,dtype=torch.float32)
            teachers_logits_res=teachers_logits_res.to(proxy.device)

            backbone_outputs,pred_speed_and_curvatur,pooled_features = proxy(inputs,labels)
            pred_speed_and_curvatur[:, :, 0] = pred_speed_and_curvatur[:, :, 0] * (speed_max - speed_min) + speed_min
            pred_speed_and_curvatur[:, :, 1] = pred_speed_and_curvatur[:, :, 1] * curv_std + curv_mean

            future_speed_and_curvatur[:, :, 0] = future_speed_and_curvatur[:, :, 0] * (speed_max - speed_min) + speed_min
            future_speed_and_curvatur[:, :, 1] = future_speed_and_curvatur[:, :, 1] * curv_std + curv_mean

            ce_loss = backbone_outputs.loss
            speed_mse_loss=torch.nn.functional.mse_loss(pred_speed_and_curvatur[:,:,0], future_speed_and_curvatur[:,:,0],reduction='mean')
            curv_mse_loss=torch.nn.functional.mse_loss(pred_speed_and_curvatur[:,:,1], future_speed_and_curvatur[:,:,1],reduction='mean')
            mse_loss=proxy.loss_fn(speed_mse_loss,curv_mse_loss)


            teachers_logits=teachers_logits_res+future_speed_and_curvatur
            res_loss= torch.nn.functional.mse_loss(teachers_logits, pred_speed_and_curvatur,reduction='mean')
            distil_loss = res_loss
            loss = 0.1 * distil_loss + 0.9 * mse_loss

            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss)
            avg_loss /= dist.get_world_size()

            if (avg_loss < best_train_loss) and (step%args.gcs==0):
                dist.barrier()
                _=save_filtered_params(args=args,model=proxy.module, file_name='best', saved_loss=avg_loss, outer_round=epoch, inner_round=step,saved_params_dict=this_prev_edge_w,check_path=check_path,local_client_data_length=train_dest.__len__(),personal_params_for_recover=personal_params_for_recover)
                best_train_loss = avg_loss

                dist.barrier()
            if dist.get_rank() == 0:
                epoch_loss+=avg_loss
                progress_bar.set_description(f'{communication}-th communication of {args.cloud_peft} ### Cloud | {epoch}-th epochs | {step}-th step | Loss: {avg_loss}')
            proxy.backward(loss)
            proxy.step()
            if step % 8 == 0:
                scheduler.step()

        if dist.get_rank()==0:
            epoch_loss=epoch_loss/(step+1)
        if epoch==args.cloud_epochs-1:
            this_prev_cloud_w=save_filtered_params(args=args,model=proxy.module,file_name=f'received_w_C{communication}',saved_loss=epoch_loss,outer_round=communication,inner_round=epoch,saved_params_dict=this_prev_edge_w,check_path=check_path,local_client_data_length=train_dest.__len__(),personal_params_for_recover=personal_params_for_recover)



    return {
        'this_prev_edge_w':this_prev_cloud_w,
        'this_prev_edge_data_length':train_dest.__len__()
    }




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help='Used by deepspeed')
    parser.add_argument('--edge_bs', type=int, default=4, help="cloud batch size: B")
    parser.add_argument('--gcs', type=int, default=8, help="gradient_accumulation_steps")
    parser.add_argument('--gpus', type=int, default=7)
    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--cloud_peft', type=str, default='lora',choices=['lora', 'perTucker','Tlora','SperTucker','perTucker-V'])
    parser.add_argument('--peft', type=str, default='lora', choices=['lora', 'perTucker','Tlora','SperTucker','perTucker-V'])
    parser.add_argument('--edge_model_path',type=str,default=r'/data/gm/DoubleCap/models/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--epochs', type=int, default=1, help="number of rounds of training")
    parser.add_argument('--data_path', type=str,default=r'/data/gm/DoubleCap/NuScenes/data')

    parser.add_argument('--client_idx', type=int, default=4, help="cloud batch size: B")
    parser.add_argument('--edge_alpha', type=float, default=0.3, help='learning rate')
    parser.add_argument('--d2edge_lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--seed_num', type=int, default=42)

    parser.add_argument('--cloud_epochs', type=int, default=4, help="number of rounds of training")
    parser.add_argument('--aggregation', type=str, default='doublecap')


    parser.add_argument('--prev_edge_w_path', type=str,default=None)
    parser.add_argument('--train_input_file', type=str,default=r'/data/gm/DoubleCap/NuScenes/data')
    parser.add_argument('--check_path', type=str,default=r'/data/gm/DoubleCap/NuScenes/data')
    parser.add_argument('--communication', type=int, default=4, help="cloud batch size: B")
    parser.add_argument('--communications', type=int, default=10, help="number of rounds of training")


    parser.add_argument('--logit_w', type=float, default=1, help='learning rate')
    parser.add_argument('--ce_w', type=float, default=1, help='learning rate')
    parser.add_argument('--T', type=float, default=1, help='learning rate')
    parser.add_argument('--struct_w', type=float, default=1, help='learning rate')

    parser.add_argument('--use_module',type=bool, default=True)

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    ds_config = {
        'train_micro_batch_size_per_gpu':args.edge_bs,
        'train_batch_size':args.edge_bs* args.gcs * args.gpus,
        "gradient_accumulation_steps": args.gcs,
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": False,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
            # "auto_scale": False
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True  # 启用内存锁定
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "stage3_max_live_parameters": 3e6,  # 减少活跃参数上限
            "stage3_max_reuse_distance": 3e6,  # 减少重用距离
            "stage3_prefetch_bucket_size": 2e7,  # 缩小预取桶大小
            "stage3_param_persistence_threshold": 1e4,  # 降低持久化阈值
        }
    }
    setup_seed(args.seed_num)
    result_json = cloud_distill2_edge(ds_config,args,args.prev_edge_w_path,args.train_input_file,args.check_path,args.communication)
    print(result_json['this_prev_edge_data_length'])

