from tqdm import tqdm
import torch
from lora_model import DriveMLLM
from utd_dataset import MultiFrameDataset
from torch.utils.data import DataLoader
import os
from training_utils import test_personal_generalize_metrics,update_cloud_train_data



def local_update(args,proxy,train_dataloader,optimizer,scheduler,check_path,node_name,saved_params_dict,communication,local_data_length,local_start_time,train_input_file,client_idx,is_test=True,batch_size=1,device=None,criterion=None,update_cloud=False,stage=1):
    best_train_loss=1000
    progress_bar = tqdm(range(args.epochs), total=args.epochs*len(train_dataloader))
    for epoch in progress_bar:
        epoch_loss = 0
        for step, (color, depth, inertial, skeleton, target,prompt_template,trainFlag,selected_idx) in enumerate(train_dataloader):
            optimizer.zero_grad()

            inputData = [torch.unsqueeze(depth, 1).cuda().float(), inertial.permute(0, 2, 1).cuda().float(),skeleton.cuda().float()]
            target = target.cuda().long()

            logits = proxy(inputData,trainFlag,prompt_template,device)

            loss=criterion(logits,target)

            if (loss < best_train_loss) and (step%args.gcs==0):
                _=save_filtered_params(args=args,model=proxy, file_name='best', saved_loss=loss, outer_round=epoch, inner_round=step,saved_params_dict=saved_params_dict,check_path=check_path,local_client_data_length=local_data_length)
                best_train_loss = loss

            epoch_loss+=loss
            progress_bar.set_description(f'{args.aggregation}: {communication}-th communication of Local update ### {node_name} | {epoch}-th epochs | {step}-th step | Loss: {loss}')
            progress_bar.update(1)
            loss.backward()
            optimizer.step()

        epoch_loss=epoch_loss/(step+1)

        if epoch==args.epochs-1:
            current_peft_weight=save_filtered_params(args=args,model=proxy,file_name=f'local_w',saved_loss=epoch_loss,outer_round=communication,inner_round=epoch,saved_params_dict=saved_params_dict,check_path=check_path,local_client_data_length=local_data_length)
            if update_cloud:
                update_cloud_train_data(args,proxy,train_input_file,client_idx,check_path,communication,device,trainFlag=trainFlag,return_logits=True,return_feat=True,return_global=False,return_personal=True,return_w=False)
            if is_test:
                test_personal_generalize_metrics(args, proxy, train_input_file, check_path, communication, node_name,batch_size,device)
            return current_peft_weight,proxy



def save_filtered_params(args,model,file_name,saved_loss,outer_round,inner_round,saved_params_dict,check_path,local_client_data_length,personal_params_for_recover=None):
    saved_state = {}


    for name, param in model.named_parameters():
        if name in list(saved_params_dict.keys()):
            if param.shape!=torch.Size([0]):
                saved_state[name] = param.detach().cpu()
            else:
                print(f'cannot obtain {name} params from updated model')

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
    return saved_state



def initial_local_model(args,prev_edge_w_path,communication,lr,pretrained_model_path,peft,device,return_training_components=True,reduction=False):

    proxy = DriveMLLM(rank=args.rank, lora_alpha=args.lora_alpha, peft=peft.lower(),pretrained_model_path=pretrained_model_path).to(device)
    criterion =torch.nn.CrossEntropyLoss().to(device)
    if prev_edge_w_path is not None:
        this_prev_edge_w=torch.load(prev_edge_w_path)['model']
        proxy.load_state_dict(this_prev_edge_w, strict=False)
    # else:
    this_prev_edge_w = {param_name: None for param_name in proxy._get_all_peft_params(proxy).keys()}
    # freeze global training local
    if (args.aggregation.lower() == 'doublecap'):
        if peft.lower=='pertucker':
            proxy.freeze_block(proxy, freeze_personal_block=False, freeze_global_block=True)

    if return_training_components:
        model_and_loss_params = list(filter(lambda p: p.requires_grad, proxy.model.parameters()))
        optimizer = torch.optim.Adam(model_and_loss_params,lr=lr, betas=(0.9, 0.999), eps=1e-8)
        scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1,verbose=False)

        return proxy,optimizer,scheduler,this_prev_edge_w,criterion
    else:
        return proxy,criterion,this_prev_edge_w


def initial_local_train_loader(proxy,train_input_file,bs):
    train_dest = MultiFrameDataset(input_file=train_input_file)
    train_dataloader = DataLoader(train_dest, shuffle=True, batch_size=bs,num_workers=0, collate_fn=train_dest.custom_collate_fn)
    this_prev_edge_data_length = train_dest.__len__()
    return train_dataloader,this_prev_edge_data_length


