#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dev='cuda'
import warnings
import json
import sys
import math
from datetime import datetime
from pathlib import Path
import torch
from options import args_parser
from utils import setup_seed
from generate_distributed_data import data_processor
import local_update_for_one_edge
import distill2edge
import distill2cloud
import gc


def generated_client_info_json(args,node_name,start_communication):
    save_root_path, base_method_name = check_dir_name()
    check_path = os.path.join(save_root_path, os.path.join(base_method_name, node_name))
    if not os.path.exists(os.path.join(check_path, 'checkpoint')):
        os.makedirs(os.path.join(check_path, 'checkpoint'))
    if not os.path.exists(os.path.join(check_path,'Test')):
        os.makedirs(os.path.join(check_path,'Test'))
    if not os.path.exists(os.path.join(check_path,'distil_cloud_train_data')):
        os.makedirs(os.path.join(check_path,'distil_cloud_train_data'))
    with open(os.path.join(check_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    return check_path

def generate_edge_model_path(args):
    large_edge_num=int(args.client_num*args.edge_model_large_frac)
    base_edge_num=args.client_num-large_edge_num
    large_edge_model_path=[args.large_model_path for _ in range(large_edge_num)]
    base_edge_model_path=[args.base_model_path for _ in range(base_edge_num)]
    args.edge_model_path=large_edge_model_path+base_edge_model_path

def check_dir_name():
    generate_edge_model_path(args)
    args.cloud_model_path = args.large_model_path if args.cloud_model_type == 'large' else args.base_model_path
    save_dir_path = os.path.join(os.path.dirname(sys.argv[0]), args.result_dir_name)


    data_info=f'C-{args.communications}_PR-{args.participate_data_ratio}_IID-{args.iid}_{args.data_type}' if args.ModalityMixFrac[2]==1 else f'C-{args.communications}_PR-{args.participate_data_ratio}_IID-{args.iid}_{"-".join([str(args.ModalityMixFrac[i]) for i in range(len(args.ModalityMixFrac))])}_{args.data_type}'
    save_dir_path = os.path.join(save_dir_path, data_info)

    backbone_name=f'largeFrac-{args.edge_model_large_frac}'

    para_name =  f'C{args.client_num}-rank{args.rank}-Cep{args.cloud_epochs}-dClr{args.d2cloud_lr}-dElr{args.d2edge_lr}-Elr{str(args.edge_lr)}'
    method_name = f'{args.aggregation}-E_{args.peft}-C_{args.cloud_peft}_{args.edge_alpha}'
    save_root_path = os.path.join(save_dir_path, '{}_{}'.format(backbone_name, para_name))
    return save_root_path, method_name

def cosine_annealing(communication, initial_lr=1e-5,min_lr=1e-5, total_communication=100):
    cosine_value=math.cos(math.pi*communication/total_communication)
    return min_lr+0.5*(initial_lr-min_lr)*(1+cosine_value)

def get_training_components(model,lr):
    model_and_loss_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    # model_and_loss_params = list(filter(lambda p: p.requires_grad, proxy.model.parameters()))+list(filter(lambda p:p.requires_grad,proxy.global_head.parameters()))+list(proxy.loss_fn.parameters())
    optimizer = torch.optim.Adam(model_and_loss_params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)
    return optimizer,scheduler

if __name__ == '__main__':
    args, argsDict = args_parser()

    warnings.filterwarnings("ignore")
    subnetFlagMap = ['D', 'I', 'S']
    setup_seed()

    root_path = os.path.join(Path(os.getcwd()),'data',f'ClientNum-{args.client_num}_IID-{args.iid}_ModalityFrac-{"-".join([str(args.ModalityMixFrac[i]) for i in range(len(args.ModalityMixFrac))])}',args.data_type) if args.data_type.lower().strip() == 'full' \
        else os.path.join(Path(os.getcwd()),'data',f'ClientNum-{args.client_num}_IID-{args.iid}_ModalityFrac-{"-".join([str(args.ModalityMixFrac[i]) for i in range(len(args.ModalityMixFrac))])}',f'{args.data_type}-{args.communications}')

    edge_check_path = {f'Client{client_idx}': None for client_idx in range(args.client_num)}
    start_communication = 0
    end_communication = args.communications


    edge_check_path['Cloud'] = generated_client_info_json(args, node_name='Cloud',start_communication=start_communication)
    exp_dirname = os.path.dirname(edge_check_path['Cloud'])
    for client_idx in range(args.client_num):
        edge_check_path[f'Client{client_idx}'] = generated_client_info_json(args, node_name=f'Client{client_idx}',start_communication=start_communication)


    for communication in range(args.startC,args.endC):
        edge_lr = cosine_annealing(communication, args.edge_lr, min_lr=args.min_lr)
        for client_idx in range(args.client_num):

            prev_edge_w_path=None if communication==0 else os.path.join(edge_check_path[f'Client{client_idx}'], 'checkpoint',f'local_w.pth')
            proxy, criterion, this_prev_edge_w = local_update_for_one_edge.initial_local_model(args,prev_edge_w_path=prev_edge_w_path,communication=communication,lr=args.edge_lr,pretrained_model_path=args.edge_model_path[client_idx],peft=args.peft,device=dev,return_training_components=False)
            proxy.print_trainable_peft_param()
            train_input_file = os.path.join(root_path, f'Client{client_idx}','train.pt') if args.data_type.lower().strip() == 'full' \
                else os.path.join(root_path, f'Client{client_idx}', f'{communication}_train.pt')
            if communication>0:
                # print(f'#################### {args.aggregation}: start {communication-1}-th round D2Edge aggregation ####################')
                train_input_cloud_file = os.path.join(root_path, 'Cloud','train.pt') if args.data_type.lower().strip() == 'full' \
                    else os.path.join(root_path, f'Cloud', f'{communication-1}_train.pt')
                check_path = edge_check_path[f'Client{client_idx}']

                train_dataloader, this_prev_edge_data_length = distill2edge.initial_d2edge_train_loader(train_input_cloud_file, train_input_file,args.edge_bs, client_idx, check_path, communication)


                d2edge_lr = cosine_annealing(communication, args.d2edge_lr, min_lr=args.min_lr)
                if args.peft.lower() =='pertucker':
                    personal_params_for_recover = proxy.set_personal_disable(proxy)
                    proxy.freeze_block(proxy, freeze_personal_block=True, freeze_global_block=False,freeze_shared_block=False, freeze_multimodal_fusion=False)
                    optimizer, scheduler = get_training_components(proxy.model, d2edge_lr)
                    proxy.train()
                    _ = distill2edge.d2edge_update(args, proxy, train_dataloader,optimizer,scheduler,check_path,communication, dev,criterion,client_idx)
                    proxy.set_personal_able(proxy, personal_params_for_recover)
                else:
                    proxy.freeze_block(proxy, freeze_personal_block=False, freeze_global_block=False,freeze_shared_block=False, freeze_multimodal_fusion=False)
                    optimizer, scheduler = get_training_components(proxy.model, d2edge_lr)
                    proxy.train()
                    _ = distill2edge.d2edge_update(args, proxy, train_dataloader,optimizer,scheduler,check_path,communication, dev,criterion,client_idx)



            # print(f'#################### {args.aggregation}: start training {communication}-th communication || Client{client_idx} ###################')

            check_path = edge_check_path[f'Client{client_idx}']


            train_dataloader, this_prev_edge_data_length = local_update_for_one_edge.initial_local_train_loader(proxy, train_input_file, args.edge_bs)


            # 训练multimodal
            proxy.freeze_block(proxy, freeze_personal_block=True, freeze_global_block=True,freeze_shared_block=True, freeze_multimodal_fusion=False)
            optimizer,scheduler=get_training_components(proxy.model,edge_lr)
            proxy.train()
            _ = local_update_for_one_edge.local_update(args, proxy, train_dataloader=train_dataloader, optimizer=optimizer,scheduler=scheduler,
                             check_path=check_path,node_name=f'Client{client_idx}',saved_params_dict=this_prev_edge_w,communication=communication,local_data_length=this_prev_edge_data_length,local_start_time=datetime.now(),
                             train_input_file=train_input_file,client_idx=client_idx, is_test=False, device=dev, criterion=criterion,update_cloud=False)
            # 微调text
            if args.peft.lower() =='pertucker':
                freeze_shared_block = False if communication == 0 else True
                proxy.freeze_block(proxy, freeze_personal_block=False, freeze_global_block=True,freeze_shared_block=freeze_shared_block, freeze_multimodal_fusion=True)
            else:
                proxy.freeze_block(proxy, freeze_personal_block=False, freeze_global_block=False,freeze_shared_block=False, freeze_multimodal_fusion=True)
            optimizer,scheduler=get_training_components(proxy.model,edge_lr)
            proxy.train()
            _ = local_update_for_one_edge.local_update(args, proxy, train_dataloader=train_dataloader, optimizer=optimizer,scheduler=scheduler,
                             check_path=check_path,node_name=f'Client{client_idx}',saved_params_dict=this_prev_edge_w,communication=communication,local_data_length=this_prev_edge_data_length,local_start_time=datetime.now(),
                             train_input_file=train_input_file,client_idx=client_idx, is_test=True, device=dev, criterion=criterion,update_cloud=True)

            del proxy, optimizer, scheduler, train_dataloader, criterion
            torch.cuda.empty_cache()
            gc.collect()

        train_input_file = os.path.join(root_path, 'Cloud','train.pt') if args.data_type.lower().strip() == 'full' \
            else os.path.join(root_path, f'Cloud', f'{communication}_train.pt')
        check_path = edge_check_path['Cloud']
        prev_cloud_w_path=os.path.join(edge_check_path['Cloud'],'checkpoint',f'cloud_w.pth') if communication>0 else None

        d2cloud_lr = cosine_annealing(communication + 1, args.d2cloud_lr, min_lr=args.min_lr)

        proxy, optimizer, scheduler, this_prev_cloud_w, criterion=distill2cloud.init_cloud_model(args, prev_cloud_w_path, lr= d2cloud_lr, pretrained_model_path=args.cloud_model_path, peft=args.cloud_peft, device=dev, return_training_components = True)
        modality_dataloader = distill2cloud.initial_cloud_train_loader(train_input_file,check_path,client_num=args.client_num,bs=args.edge_bs,communication=communication)
        proxy.train()

        _=distill2cloud.edges_distill2_cloud(args,proxy,modality_dataloader,optimizer,criterion,this_prev_cloud_w,train_input_file,check_path,communication,dev,is_test=True)
        del proxy, optimizer, scheduler, modality_dataloader, criterion
        torch.cuda.empty_cache()
        gc.collect()
