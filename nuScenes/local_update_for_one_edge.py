import deepspeed.comm as dist
from tqdm import tqdm
from datetime import datetime
import torch
from lora_model import DriveMLLM
from deepspeed.ops.adam import DeepSpeedCPUAdam
from nuscenes_dataset import MultiFrameDataset
from torch.utils.data import DistributedSampler,DataLoader
from utils import setup_seed
import os
import deepspeed
import argparse
from training_utils import test_personal_generalize_metrics,update_cloud_train_data



def local_update(args,proxy,train_dataloader,scheduler,sampler,check_path,node_name,saved_params_dict,communication,local_data_length,local_start_time,train_input_file,client_idx,is_test=True,batch_size=1,bfloat=False):
    best_train_loss=1000
    # speed_std = 1.8046681286146409
    # speed_mean = 2.5373214399490234
    speed_min =0
    speed_max = 10.082848494377343
    curv_std = 0.07511833558384672
    curv_mean = 0.021407328091665685


    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        epoch_start_time = datetime.now()
        epoch_loss = 0
        epoch_ce_loss, epoch_mse_loss,epoch_speed_mse_loss,epoch_curv_mse_loss = 0, 0,0,0

        progress_bar = tqdm(enumerate(train_dataloader),total=len(train_dataloader))
        for step, (inputs, labels, selected_idx,future_speed_and_curvatur,future_trajs) in progress_bar:
            step_start_time=datetime.now()

            inputs = {
                k: v.to(proxy.device).requires_grad_(True) if k == "pixel_values" else v.to(proxy.device)
                for k, v in inputs.items()
            }

            labels = labels.to(proxy.device)
            future_speed_and_curvatur = torch.tensor(future_speed_and_curvatur, device=proxy.device,dtype=torch.bfloat16) if bfloat else torch.tensor(future_speed_and_curvatur,device=proxy.device,dtype=torch.float32)

            # inputs['labels']=labels
            backbone_outputs,pred_speed_and_curvatur,pooled_features = proxy(inputs,labels)

            pred_speed_and_curvatur[:, :, 0] = pred_speed_and_curvatur[:, :, 0] * (speed_max-speed_min) + speed_min
            pred_speed_and_curvatur[:, :, 1] = pred_speed_and_curvatur[:, :, 1] * curv_std + curv_mean

            future_speed_and_curvatur[:, :, 0] = future_speed_and_curvatur[:, :, 0] * (speed_max-speed_min) + speed_min
            future_speed_and_curvatur[:, :, 1] = future_speed_and_curvatur[:, :, 1] * curv_std + curv_mean
            # pred_speed_and_curvatur = pred_speed_and_curvatur.view(-1, 10, 2)

            speed_mse_loss=torch.nn.functional.mse_loss(pred_speed_and_curvatur[:,:,0], future_speed_and_curvatur[:,:,0],reduction='mean')
            curv_mse_loss=torch.nn.functional.mse_loss(pred_speed_and_curvatur[:,:,1], future_speed_and_curvatur[:,:,1],reduction='mean')
            ce_loss = backbone_outputs.loss

            if bfloat:
                speed_mse_loss=speed_mse_loss.to(torch.bfloat16)
                curv_mse_loss=curv_mse_loss.to(torch.bfloat16)
                ce_loss=ce_loss.to(torch.bfloat16)

            mse_loss=proxy.loss_fn(speed_mse_loss,curv_mse_loss)

            if bfloat:
                mse_loss=mse_loss.to(torch.bfloat16)

            loss = args.lambda_lm * ce_loss + (1 - args.lambda_lm) * mse_loss
            if bfloat:

                loss=loss.to(torch.bfloat16)

            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss)
            avg_loss /= dist.get_world_size()

            if (avg_loss < best_train_loss) and (step%args.gcs==0):
                tmp_start=datetime.now()
                dist.barrier()
                _=save_filtered_params(args=args,model=proxy.module, file_name='best', saved_loss=avg_loss, outer_round=epoch, inner_round=step,saved_params_dict=saved_params_dict,check_path=check_path,local_client_data_length=local_data_length)
                best_train_loss = avg_loss

                dist.barrier()
            if dist.get_rank() == 0:

                epoch_loss+=avg_loss

                progress_bar.set_description(f'{communication}-th communication of {args.peft} ### {node_name} | {epoch}-th epochs | {step}-th step | Loss: {avg_loss}')
            proxy.backward(loss)
            proxy.step()
            if step%8==0:
                scheduler.step()

        if dist.get_rank()==0:
            epoch_loss=epoch_loss/(step+1)

        if epoch==args.epochs-1:
            current_peft_weight=save_filtered_params(args=args,model=proxy.module,file_name=f'local_w_C{communication}',saved_loss=epoch_loss,outer_round=communication,inner_round=epoch,saved_params_dict=saved_params_dict,check_path=check_path,local_client_data_length=local_data_length)
            update_cloud_train_data(args,proxy,train_input_file,client_idx,check_path,communication,return_logits=True,return_feat=True,return_global=False,return_personal=True,return_w=False)

            if is_test:
                test_personal_generalize_metrics(args, proxy, train_input_file, check_path, communication, node_name,batch_size)
            return current_peft_weight,proxy



def initial_local_model(args,prev_edge_w_path,ds_config,communication=0):

    proxy = DriveMLLM(rank=args.rank, lora_alpha=args.lora_alpha, peft=args.peft.lower(),pretrained_model_path=args.edge_model_path)

    if prev_edge_w_path is not None:
        this_prev_edge_w=torch.load(prev_edge_w_path,weights_only=False)['model']
        proxy.load_state_dict(this_prev_edge_w, strict=False)

    this_prev_edge_w = {param_name: None for param_name in proxy._get_all_peft_params(proxy).keys()}
    if args.peft.lower() =='pertucker':
        proxy.freeze_block(proxy, freeze_personal_block=False, freeze_global_block=True)

    model_and_loss_params = list(filter(lambda p: p.requires_grad, proxy.model.parameters())) + list(proxy.loss_fn.parameters())
    optimizer = DeepSpeedCPUAdam(model_and_loss_params,lr=args.edge_lr, betas=(0.9, 0.999), eps=1e-8)
    proxy, optimizer, _, scheduler = deepspeed.initialize(
        model=proxy,
        optimizer=optimizer,
        model_parameters=model_and_loss_params,
        # training_data=train_dset,
        config_params=ds_config,
        lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
    )

    return proxy,optimizer,scheduler,this_prev_edge_w



def initial_local_train_loader(proxy,train_input_file):
    train_dest = MultiFrameDataset(sample_root=args.data_path, input_file=train_input_file,processor=proxy.processor, model_name=proxy.model_name)
    sampler = DistributedSampler(train_dest, num_replicas=args.gpus, rank=torch.distributed.get_rank())
    train_dataloader = DataLoader(train_dest, shuffle=False, batch_size=args.edge_bs,sampler=sampler, num_workers=0, collate_fn=train_dest.custom_collate_fn)
    this_prev_edge_data_length = train_dest.__len__()
    return train_dataloader,sampler,this_prev_edge_data_length



def create_local_trainer(ds_config,args,prev_edge_w_path,client_idx,train_input_file,check_path,communication):
    local_start_time=datetime.now()

    proxy, optimizer, scheduler,this_prev_edge_w=initial_local_model(args,prev_edge_w_path,ds_config,communication)

    train_dataloader, sampler, this_prev_edge_data_length=initial_local_train_loader(proxy,train_input_file)


    if communication<args.communications:
        proxy.train()
        is_test = True if communication==args.communications-1 else False
        this_prev_edge_w,_ = local_update(args, proxy, train_dataloader=train_dataloader, scheduler=scheduler, sampler=sampler,
                                        check_path=check_path,
                                        node_name=f'Client{client_idx}',
                                        saved_params_dict=this_prev_edge_w,
                                        communication=communication,
                                        local_data_length=this_prev_edge_data_length,
                                        local_start_time=local_start_time,
                                        train_input_file=train_input_file,
                                        client_idx=client_idx,batch_size=args.edge_bs,is_test=is_test)


def save_filtered_params(args,model,file_name,saved_loss,outer_round,inner_round,saved_params_dict,check_path,local_client_data_length,personal_params_for_recover=None):
    saved_state = {}
    gather_param=[]
    gather_param_name=[]

    for name, param in model.named_parameters():
        if name in list(saved_params_dict.keys()):
            if param.shape!=torch.Size([0]):
                saved_state[name] = param.detach().cpu()
            else:
                gather_param.append(param)
                gather_param_name.append(name)
    if len(gather_param)!=0:
        with deepspeed.zero.GatheredParameters(gather_param, modifier_rank=0):
            for idx, param in enumerate(gather_param):
                name = gather_param_name[idx]
                saved_state[name] = param.detach().cpu()


    if dist.get_rank()==0:
        if personal_params_for_recover is not None:
            for name in personal_params_for_recover.keys():
                saved_state[name] = personal_params_for_recover[name]
        torch.save({
            'args':args,
            'model':saved_state,
            'loss':saved_loss,
            'epoch':outer_round,
            'step':inner_round,
            'data_len':local_client_data_length
        },os.path.join(check_path,'checkpoint',f'{file_name}.pth'))
        saved_path=os.path.join(check_path,'checkpoint',f'{file_name}.pth')
        print(f"###################### saved at : {saved_path}")
    return saved_state


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud_epochs', type=int, default=4, help="number of rounds of training")
    parser.add_argument('--data_type', type=str, default='nointersection',choices=['full', 'nointersection', 'incremental'])
    parser.add_argument('--raw_data_path', type=str, default=r'/data/gm/DoubleCap/NuScenes/data')
    parser.add_argument('--lambda_lm', type=float, default=0.3, help='learning rate')
    parser.add_argument('--local_rank', type=int, default=-1, help='Used by deepspeed')
    parser.add_argument('--epochs', type=int, default=1, help="number of rounds of training")
    parser.add_argument('--communications', type=int, default=10, help="number of rounds of training")
    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--peft', type=str, default='lora',choices=['lora', 'perTucker','Tlora'])
    parser.add_argument('--data_path', type=str,default=r'/data/gm/DoubleCap/NuScenes/data_resize0.25')
    parser.add_argument('--aggregation', type=str, default='doublecap')
    parser.add_argument('--gcs', type=int, default=8, help="gradient_accumulation_steps")
    parser.add_argument('--gpus', type=int, default=7)
    parser.add_argument('--edge_lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--root_path', type=str,
                        default=r'/data/gm/DoubleCap/models/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--client_num', type=int, default=2, help='number of users')

    parser.add_argument('--edge_model_path',type=str,default=r'/data/gm/DoubleCap/models/Qwen2-VL-2B-Instruct')
    parser.add_argument('--edge_bs', type=int, default=2, help="cloud batch size: B")
    parser.add_argument('--seed_num', type=int, default=42)


    parser.add_argument('--prev_edge_w_path', type=str,default=None)
    parser.add_argument('--train_input_file', type=str,default=r'/data/gm/DoubleCap/NuScenes/data')
    parser.add_argument('--check_path', type=str,default=r'/data/gm/DoubleCap/NuScenes/data')
    parser.add_argument('--communication', type=int, default=4)
    parser.add_argument('--client_idx', type=int, default=None)

    parser.add_argument('--stage', type=int, default=1,choices=[1,2])
    parser.add_argument('--prev_cloud_w_path', type=str, default=None)
    parser.add_argument('--only_test',  type=bool, default=False)
    parser.add_argument('--generate_teacher', type=bool, default=False)

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
    setup_seed()

    create_local_trainer(ds_config,args,args.prev_edge_w_path,args.client_idx,args.train_input_file,args.check_path,args.communication)
