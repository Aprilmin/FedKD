from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from utd_dataset import Cloud2EdgesMultiFrameDataset
import torch
import os


def initial_d2edge_train_loader(train_input_cloud_file,train_input_edge_file,bs,client_idx,check_path,communication):
    trainFlag = torch.load(train_input_edge_file)['trainFlag']
    edge_trainFlag_str = '-'.join([str(int(trainFlag[i])) for i in range(len(trainFlag))])
    node_data= torch.load(train_input_cloud_file)

    cloud_data = node_data['data'][edge_trainFlag_str]

    prompt_template = node_data['prompt_template']
    teacher_data=torch.load(os.path.join(check_path.replace(f'Client{client_idx}', 'Cloud'), 'distil_cloud_train_data',f'{communication - 1}_train.pt'))[edge_trainFlag_str]['data']


    train_dest = Cloud2EdgesMultiFrameDataset(cloud_data, trainFlag, prompt_template, teacher_data)
    train_dataloader = DataLoader(train_dest, shuffle=True, batch_size=bs,num_workers=0, collate_fn=train_dest.custom_collate_fn)
    this_prev_edge_data_length = train_dest.__len__()
    return train_dataloader,this_prev_edge_data_length


def d2edge_update(args, proxy, train_dataloader,optimizer,scheduler,check_path,communication, device,criterion,client_idx):
    progress_bar = tqdm(range(args.d2edge_epochs), total=args.d2edge_epochs*len(train_dataloader))
    for epoch in range(args.d2edge_epochs):
        for step, (color, depth,inertial ,skeleton , target,prompt_template,trainFlag,selected_idx,teachers_logit) in enumerate(train_dataloader):

            optimizer.zero_grad()
            step_start_time=datetime.now()
            inputData = [torch.unsqueeze(depth, 1).cuda().float(), inertial.permute(0, 2, 1).cuda().float(),skeleton.cuda().float()]
            target = target.cuda().long()
            teachers_logit = teachers_logit.cuda()

            logits = proxy(inputData,trainFlag,prompt_template,device)
            task_loss=criterion(logits,target)
            T = 4.0
            soft_labels = torch.nn.functional.softmax(teachers_logit / T, dim=1)
            student_log_probs = torch.nn.functional.log_softmax(logits / T, dim=1)

            kl_loss = torch.nn.functional.kl_div(student_log_probs, soft_labels, reduction="batchmean") * (T ** 2)
            loss=args.edge_alpha * kl_loss + (1 - args.edge_alpha) * task_loss



            progress_bar.set_description(f'{args.aggregation}: {communication}-th communication of {args.cloud_peft} ### d2Edge at {client_idx}-th Edge | {epoch}-th epochs | {step}-th step | Loss: {loss}')
            progress_bar.update(1)
            loss.backward()
            optimizer.step()

    return proxy


