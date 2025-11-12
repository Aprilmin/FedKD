from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from lora_model import Edge2CloudDistil
from utd_dataset import Edges2CloudMultiFrameDataset
from local_update_for_one_edge import save_filtered_params
from training_utils import test_personal_generalize_metrics,update_cloud_train_data2edge

import numpy as np
import torch

import os


def init_cloud_model(args, prev_cloud_w_path, lr, pretrained_model_path, peft, device,return_training_components=True):
    proxy = Edge2CloudDistil(rank=args.rank, lora_alpha=args.lora_alpha, peft=peft,
                             pretrained_model_path=pretrained_model_path, client_num=args.client_num,
                             lambda_grl=1.0).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    if prev_cloud_w_path is not None:
        this_prev_cloud_w = torch.load(prev_cloud_w_path)['model']
        proxy.load_state_dict(this_prev_cloud_w, strict=False)

    this_prev_cloud_w = {param_name: None for param_name in proxy._get_all_peft_params(proxy).keys()}
    if return_training_components:
        param_to_optimize = list(filter(lambda p: p.requires_grad, proxy.model.parameters())) + list(proxy.discriminator.parameters())+list(proxy.loss_fn.parameters())

        optimizer = torch.optim.Adam(param_to_optimize, lr=lr, betas=(0.9, 0.999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)
        return proxy, optimizer, scheduler, this_prev_cloud_w, criterion
    else:
        return proxy, criterion, this_prev_cloud_w


def initial_cloud_train_loader(train_input_file,check_path,client_num,bs,communication):
    node_data = torch.load(train_input_file)
    cloud_train_data = node_data['data']

    modality_teachers_list={modality_name:[] for modality_name in cloud_train_data.keys()}

    for client_idx in range(client_num):
        client_input_file_path = os.path.join(check_path.replace('Cloud', f'Client{client_idx}'),'distil_cloud_train_data', f'{communication}_train.pt')
        teacher_data=torch.load(client_input_file_path, map_location='cpu')
        modality_teachers_list[teacher_data['trainFlag']].append({'teacher_data':teacher_data['data'],'teacher_class':client_idx})

    modality_dataloader={}
    for modality_name in modality_teachers_list.keys():

        train_dest=Edges2CloudMultiFrameDataset(cloud_data=cloud_train_data[modality_name],teacher_data=modality_teachers_list[modality_name],prompt_template=node_data['prompt_template'],trainFlag=np.array(modality_name.split('-'),dtype=np.float32))
        modality_dataloader[modality_name]={
            'dataloader':DataLoader(train_dest, shuffle=True, batch_size=bs,num_workers=0, collate_fn=train_dest.custom_collate_fn),
            'this_prev_edge_data_length':train_dest.__len__()
        }

    return modality_dataloader


def edges_distill2_cloud(args,proxy,modality_dataloader,optimizer,criterion,this_prev_cloud_w,train_input_file,check_path,communication,device,is_test=False):
    cloud_start_time = datetime.now()
    best_train_loss=1000000
    kl_warmup_steps=5
    progress_bar = tqdm(range(args.cloud_epochs))
    for epoch in range(args.cloud_epochs):
        epoch_loss = 0
        local_client_data_length=0
        for modality_name in modality_dataloader.keys():
            train_dataloader = modality_dataloader[modality_name]['dataloader']
            local_client_data_length += modality_dataloader[modality_name]['this_prev_edge_data_length']

            for step, (color, depth,inertial ,skeleton ,target,prompt_template,trainFlag,selected_idx,teacher_personal_feat,teacher_fusion_feat,teacher_text_feat,teachers_logit,teachers_class,teacher_num) in enumerate(train_dataloader):
                optimizer.zero_grad()

                inputData = [torch.unsqueeze(depth, 1).cuda().float(), inertial.permute(0, 2, 1).cuda().float(),skeleton.cuda().float()]
                target = target.cuda().long()


                batch_size=target.shape[0]
                for sample_idx in range(batch_size):
                    for teacher_idx in range(teacher_num):
                        teacher_personal_feat[sample_idx][teacher_idx]=teacher_personal_feat[sample_idx][teacher_idx].cuda()
                        teacher_fusion_feat[sample_idx][teacher_idx]=teacher_fusion_feat[sample_idx][teacher_idx].cuda()
                        teacher_text_feat[sample_idx][teacher_idx]=teacher_text_feat[sample_idx][teacher_idx].cuda()



                kl_gate=True if step>=kl_warmup_steps else False

                teacher_loss, student_entropy,fused_feat,text_feat,logits = proxy(inputData,trainFlag,prompt_template,device,return_components=False,teachers_personal_feat=teacher_personal_feat,teacher_fusion_feat=teacher_fusion_feat,teacher_text_feat=teacher_text_feat,teachers_class=teachers_class,step_ratio=step/len(train_dataloader),kl_gate=kl_gate,teacher_num=teacher_num)
                k = 5
                progress = (epoch + 1) / (args.cloud_epochs)
                alpha = 1. / (1. + np.exp(k * (progress - 0.5)))
                beta = 1. - alpha
                discriminator_loss = alpha * teacher_loss - beta * student_entropy
                discriminator_loss = discriminator_loss.to(torch.bfloat16)
                task_loss=criterion(logits,target)

                loss = 0.1 * discriminator_loss + 0.9 * task_loss

                if (loss < best_train_loss) and (step%args.gcs==0):
                    _=save_filtered_params(args=args,model=proxy, file_name='best', saved_loss=loss, outer_round=epoch, inner_round=step,saved_params_dict=this_prev_cloud_w,check_path=check_path,local_client_data_length=local_client_data_length)
                    best_train_loss = loss

                with open(os.path.join(check_path,f"{args.cloud_peft}_d2cloud_step_loss.csv"), 'a', encoding='utf-8') as f:
                    epoch_loss+=loss
                    progress_bar.set_description(f'{args.aggregation}: {communication}-th communication of {args.cloud_peft} ### d2Cloud | {epoch}-th epochs | {step}-th step | teacherLoss: {teacher_loss}, studentLoss: {student_entropy}, taskLoss:{task_loss}')

                loss.backward()
                optimizer.step()

        epoch_loss=epoch_loss/(step+1)


        if epoch==args.cloud_epochs-1:
            this_prev_cloud_w=save_filtered_params(args=args,model=proxy,file_name=f'cloud_w',saved_loss=epoch_loss,outer_round=communication,inner_round=epoch,saved_params_dict=this_prev_cloud_w,check_path=check_path,local_client_data_length=local_client_data_length)
            update_cloud_train_data2edge(args,proxy,train_input_file,client_idx=None,check_path=check_path,communication=communication,device=device,return_logits=True,return_feat=False,return_global=False,return_personal=False,return_w=False)
            if is_test:
                test_personal_generalize_metrics(args,proxy,train_input_file,check_path,communication,node_name='Cloud',batch_size=args.edge_bs,device=device)


    return {
        'this_prev_cloud_w':this_prev_cloud_w,
        'this_prev_cloud_data_length':local_client_data_length
    }







