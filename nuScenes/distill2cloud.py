from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DistributedSampler,DataLoader
from lora_model import Edge2CloudDistil
import deepspeed.comm as dist
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from nuscenes_dataset import Edges2CloudMultiFrameDataset
from local_update_for_one_edge import save_filtered_params
from training_utils import test_personal_generalize_metrics,update_cloud_train_data
from utils import setup_seed
import numpy as np
import torch
import argparse


def init_cloud_model(ds_config,args,prev_cloud_w_path):

    proxy = Edge2CloudDistil(rank=args.rank, lora_alpha=args.lora_alpha, peft=args.cloud_peft.lower(),pretrained_model_path=args.cloud_model_path,
                             client_num=args.client_num,lambda_grl = 1.0)

    if prev_cloud_w_path is not None:
        this_prev_cloud_w = torch.load(prev_cloud_w_path, weights_only=False)['model']
        proxy.load_state_dict(this_prev_cloud_w, strict=False)
    # else:
    this_prev_cloud_w = {param_name: None for param_name in proxy._get_all_peft_params(proxy).keys()}
    param_to_optimize = list(filter(lambda p: p.requires_grad, proxy.model.parameters())) + list(proxy.discriminator.parameters())  + list(proxy.loss_fn.parameters())
    # param_to_optimize=list(filter(lambda p: p.requires_grad, proxy.model.parameters()))+list(proxy.discriminator.parameters())+list(filter(lambda p:p.requires_grad,proxy.global_head.parameters()))+list(proxy.loss_fn.parameters())
    optimizer = DeepSpeedCPUAdam(param_to_optimize,lr=args.d2cloud_lr, betas=(0.9, 0.999), eps=1e-8)
    proxy, optimizer, _, scheduler = deepspeed.initialize(
        model=proxy,
        optimizer=optimizer,
        model_parameters=param_to_optimize,
        config_params=ds_config,
        lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
    )
    return proxy,optimizer,scheduler,this_prev_cloud_w

def edges_distill2_cloud(ds_config,args,prev_cloud_w_path,train_input_file,check_path,communication,bfloat):
    cloud_start_time = datetime.now()
    proxy,optimizer,scheduler,this_prev_cloud_w=init_cloud_model(ds_config,args,prev_cloud_w_path)

    train_dest = Edges2CloudMultiFrameDataset(sample_root=args.data_path, input_file=train_input_file,processor=proxy.processor, check_path=args.check_path,client_num=args.client_num, model_name=proxy.model_name)
    sampler = DistributedSampler(train_dest, num_replicas=args.gpus, rank=torch.distributed.get_rank())
    train_dataloader = DataLoader(train_dest, shuffle=False, batch_size=args.edge_bs, sampler=sampler,num_workers=0, collate_fn=train_dest.custom_collate_fn)
    proxy.train()
    best_train_loss=1000000
    kl_warmup_steps=5
    speed_std = 1.8046681286146409
    speed_mean = 2.5373214399490234
    curv_std = 0.07511833558384672
    curv_mean = 0.021407328091665685
    speed_min = 0
    speed_max = 10.082848494377343
    for epoch in range(args.cloud_epochs):
        sampler.set_epoch(epoch)

        epoch_loss = 0


        progress_bar = tqdm(enumerate(train_dataloader),total=len(train_dataloader))

        for step, (inputs,labels,teachers_res_logit, teachers_personal_features,teachers_class,future_speed_and_curvatur, future_trajs,selected_idx) in progress_bar:
            step_start_time=datetime.now()

            inputs = {
                k: v.to(proxy.device).requires_grad_(True) if k == "pixel_values" else v.to(proxy.device)
                for k, v in inputs.items()
            }
            labels = labels.to(proxy.device)
            future_speed_and_curvatur =torch.tensor(future_speed_and_curvatur,device=proxy.device,dtype=torch.bfloat16) if bfloat else torch.tensor(future_speed_and_curvatur, device=proxy.device, dtype=torch.float32)


            for sample_idx in range(len(teachers_personal_features)):
                for teacher_idx in range(len(teachers_personal_features[sample_idx])):
                    teachers_personal_features[sample_idx][teacher_idx]=teachers_personal_features[sample_idx][teacher_idx].to(proxy.device,dtype=torch.bfloat16) if bfloat else teachers_personal_features[sample_idx][teacher_idx].to(proxy.device,dtype=torch.float32)


            teachers_class=teachers_class.to(proxy.device)

            kl_gate=True if step>=kl_warmup_steps else False

            pred_speed_and_curvatur,ce_loss,kl_loss,teacher_loss, student_entropy,pooled_features= proxy(inputs,labels,teachers_personal_features,teachers_class,kl_gate,step/len(train_dataloader),bfloat=bfloat)

            pred_speed_and_curvatur[:, :, 0] = pred_speed_and_curvatur[:, :, 0] * (speed_max - speed_min) + speed_min
            pred_speed_and_curvatur[:, :, 1] = pred_speed_and_curvatur[:, :, 1] * curv_std + curv_mean

            future_speed_and_curvatur[:, :, 0] = future_speed_and_curvatur[:, :, 0] * (speed_max - speed_min) + speed_min
            future_speed_and_curvatur[:, :, 1] = future_speed_and_curvatur[:, :, 1] * curv_std + curv_mean
            speed_mse_loss=torch.nn.functional.mse_loss(pred_speed_and_curvatur[:,:,0], future_speed_and_curvatur[:,:,0],reduction='mean')
            curv_mse_loss=torch.nn.functional.mse_loss(pred_speed_and_curvatur[:,:,1], future_speed_and_curvatur[:,:,1],reduction='mean')

            mse_loss=proxy.loss_fn(speed_mse_loss,curv_mse_loss)


            if bfloat:
                speed_mse_loss=speed_mse_loss.to(torch.bfloat16)
                curv_mse_loss=curv_mse_loss.to(torch.bfloat16)
                ce_loss=ce_loss.to(torch.bfloat16)

                mse_loss=mse_loss.to(torch.bfloat16)

            task_loss =  mse_loss

            if args.use_module:
                k=5
                progress=(epoch+1)/(args.cloud_epochs)
                alpha = 1. / (1. + np.exp(k * (progress - 0.5)))
                beta = 1. - alpha
                discriminator_loss = alpha * teacher_loss - beta * student_entropy
                discriminator_loss=discriminator_loss.to(torch.bfloat16)


            else:
                discriminator_loss=0
                teacher_num=len(teachers_personal_features[0])


                for teacher_idx in range(teacher_num):
                    teacher_res_logits=[]
                    for sample_idx in range(len(teachers_personal_features)):
                        teacher_res_logits.append(teachers_res_logit[sample_idx][teacher_idx])
                    teacher_res_logits=torch.stack(teacher_res_logits).to(proxy.device)
                    teacher_logits=teacher_res_logits+future_speed_and_curvatur

                    speed_kd_loss = torch.nn.functional.mse_loss(pred_speed_and_curvatur[:, :, 0],teacher_logits[:, :, 0], reduction='mean')
                    curv_kd_loss = torch.nn.functional.mse_loss(pred_speed_and_curvatur[:, :, 1],teacher_logits[:, :, 1], reduction='mean')
                    discriminator_loss = proxy.loss_fn(speed_kd_loss ,curv_kd_loss)

            loss = 0.1 * discriminator_loss + 0.9 * task_loss
            if bfloat:
                loss=loss.to(torch.bfloat16)


            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss)
            avg_loss /= dist.get_world_size()






            if (avg_loss < best_train_loss) and (step%args.gcs==0):
                tmp_start=datetime.now()
                dist.barrier()
                _=save_filtered_params(args=args,model=proxy.module, file_name='best', saved_loss=avg_loss, outer_round=epoch, inner_round=step,saved_params_dict=this_prev_cloud_w,check_path=check_path,local_client_data_length=train_dest.__len__())
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
            this_prev_cloud_w=save_filtered_params(args=args,model=proxy.module,file_name=f'cloud_w_C{communication}',saved_loss=epoch_loss,outer_round=communication,inner_round=epoch,saved_params_dict=this_prev_cloud_w,check_path=check_path,local_client_data_length=train_dest.__len__())
            update_cloud_train_data(args,proxy,train_input_file,client_idx=None,check_path=check_path,communication=communication,return_logits=True,return_feat=False,return_global=False,return_personal=False,return_w=False)
            if communication==args.communications-1:
                test_personal_generalize_metrics(args,proxy,train_input_file,check_path,communication,node_name='Cloud',batch_size=args.edge_bs)


    return {
        'this_prev_cloud_w':this_prev_cloud_w,
        'this_prev_cloud_data_length':train_dest.__len__()
    }





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str, default=r'/data/gm/DoubleCap/NuScenes/data')
    parser.add_argument('--local_rank', type=int, default=-1, help='Used by deepspeed')
    parser.add_argument('--edge_bs', type=int, default=4, help="cloud batch size: B")
    parser.add_argument('--gcs', type=int, default=8, help="gradient_accumulation_steps")
    parser.add_argument('--gpus', type=int, default=7)
    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--cloud_peft', type=str, default='lora',choices=['lora', 'perTucker','Tlora'])
    parser.add_argument('--peft', type=str, default='lora',choices=['lora', 'perTucker', 'Tlora'])
    parser.add_argument('--cloud_model_path',type=str,default=r'/data/gm/DoubleCap/models/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--edge_model_path', type=str,default=r'/data/gm/DoubleCap/models/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--epochs', type=int, default=1, help="number of rounds of training")
    parser.add_argument('--client_num', type=int, default=2, help='number of users')
    parser.add_argument('--data_path', type=str,default=r'/data/gm/DoubleCap/NuScenes/data')
    parser.add_argument('--d2cloud_lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--d2edge_lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--d2cloud_alpha', type=float, default=0.3, help='learning rate')
    parser.add_argument('--seed_num', type=int, default=42)
    parser.add_argument('--aggregation', type=str, default='doublecap')

    parser.add_argument('--prev_cloud_w_path', type=str,default=None)
    parser.add_argument('--prev_edge_w_path', type=str,default=None)
    parser.add_argument('--train_input_file', type=str,default=r'/data/gm/DoubleCap/NuScenes/data')
    parser.add_argument('--check_path', type=str,default=r'/data/gm/DoubleCap/NuScenes/data')
    parser.add_argument('--communication', type=int, default=4, help="cloud batch size: B")
    parser.add_argument('--communications', type=int, default=10, help="number of rounds of training")


    parser.add_argument('--ib_w', type=float, default=1, help='learning rate')
    parser.add_argument('--vib_w', type=float, default=1, help='learning rate')
    parser.add_argument('--discriminator_w', type=float, default=1, help='learning rate')
    parser.add_argument('--struct_w', type=float, default=1, help='learning rate')
    parser.add_argument('--lambda_lm', type=float, default=0.01, help='learning rate')
    parser.add_argument('--use_module',type=bool, default=True)
    parser.add_argument('--cloud_epochs', type=int, default=4, help="number of rounds of training")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    ds_config = {
        'train_micro_batch_size_per_gpu':args.edge_bs,
        'train_batch_size':args.edge_bs* args.gcs * args.gpus,
        "gradient_accumulation_steps": args.gcs,
        "gradient_clipping": 1.0,
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
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "contiguous_memory_optimization": True,
            "cpu_checkpointing": True
        }
    }
    if '7B' in args.cloud_model_path:
        ds_config["bf16"] = {
            "enabled": True
        }
        bfloat=True
    else:
        ds_config["fp16"] = {
            "enabled": False,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
            # "auto_scale": False
        }
        bfloat=False
    setup_seed()
    result_json = edges_distill2_cloud(ds_config,args,args.prev_cloud_w_path,args.train_input_file,args.check_path,args.communication,bfloat)
