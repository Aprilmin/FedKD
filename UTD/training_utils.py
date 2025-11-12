import os
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from utd_dataset import MultiFrameDataset,SingleModalDataset


def test_personal_generalize_metrics(args,proxy,train_input_file,check_path,communication,node_name,batch_size,device):

    test_input_file_zoos={
        client_name:os.path.join(Path(train_input_file).parent.parent,client_name,'test.pt') for client_name in list(filter(lambda x:'cloud' not in x.lower(),os.listdir(Path(train_input_file).parent.parent)))
    }

    personal_metric=[]
    generalized_metrics=[]
    for test_data_owned_name in test_input_file_zoos.keys():


        test_metrics_for_one_dataset=test_trained_model(args,proxy,test_data_path=test_input_file_zoos[test_data_owned_name],device=device,communication=communication,node_name=node_name)
        if test_data_owned_name==node_name:
            personal_metric.append(test_metrics_for_one_dataset)
        else:
            generalized_metrics.append(test_metrics_for_one_dataset)
    generalized_metrics=np.array(generalized_metrics)
    generalized_metrics=generalized_metrics.mean(axis=0)


    if 'client' in node_name.lower():
        saved_personal_metric_csv_path=os.path.join(check_path,'Test','personal.csv')
        if not os.path.exists(saved_personal_metric_csv_path):
            with open(saved_personal_metric_csv_path,'w',encoding='utf-8') as f:
                f.write('communication,precision,accuracy,f1,inference_time\n')
        if len(personal_metric)!=0:
            with open(saved_personal_metric_csv_path, 'a', encoding='utf-8') as f:
                f.write(f'{communication},{",".join([str(personal_metric[0][i]) for i in range(len(personal_metric[0]))])}\n')
    if 'cloud' in node_name.lower():
        saved_generalized_metric_csv_path=os.path.join(check_path,'Test','generalized.csv')
        if not os.path.exists(saved_generalized_metric_csv_path):
            with open(saved_generalized_metric_csv_path,'w',encoding='utf-8') as f:
                f.write('communication,precision,accuracy,f1,inference_time\n')
        with open(saved_generalized_metric_csv_path, 'a', encoding='utf-8') as f:
            f.write(f'{communication},{",".join([str(generalized_metrics[i]) for i in range(len(generalized_metrics))])}\n')


def test_trained_model(args,proxy,test_data_path,device,communication,node_name):
    test_dest = MultiFrameDataset(input_file=test_data_path)
    test_dataloader = DataLoader(test_dest, shuffle=True, batch_size=args.edge_bs,num_workers=0, collate_fn=test_dest.custom_collate_fn)

    proxy.eval()

    with torch.no_grad():
        targetTemp, outputTemp = [], []
        start_time = datetime.now()
        for step, (color, depth, inertial, skeleton, target,prompt_template,trainFlag,selected_idx) in enumerate(test_dataloader):

            inputData = (torch.unsqueeze(depth, 1).cuda().float(), inertial.permute(0, 2, 1).cuda().float(), skeleton.cuda().float())
            target = target.cuda().long()

            output = proxy(inputData,trainFlag,prompt_template,device)

            targetTemp.extend(target.detach().cpu().numpy())
            outputTemp.extend(torch.max(output, 1)[1].detach().cpu().numpy())
        report = classification_report(targetTemp, outputTemp, output_dict=True, zero_division=0)
        inference_time=(datetime.now()-start_time)/len(targetTemp)

    return [report["weighted avg"]["precision"],report["accuracy"],report["weighted avg"]["f1-score"],inference_time]




def update_cloud_train_data(args,proxy,train_input_file,client_idx,check_path,communication,device,trainFlag,return_logits=False,return_feat=True,return_global=False,return_personal=False,return_w=False):
    public_train_path=train_input_file.replace(f'Client{client_idx}','Cloud') if client_idx is not None else train_input_file

    node_data = torch.load(public_train_path)
    data = node_data['data']
    prompt_template = node_data['prompt_template']
    edge_trainFlag_str='-'.join([str(int(trainFlag[i])) for i in range(len(trainFlag))])
    public_train_dest =SingleModalDataset(data[edge_trainFlag_str],trainFlag,prompt_template)
    local_eval_result=[dict() for _ in range(public_train_dest.__len__())]
    proxy.eval()

    public_train_dataloader = DataLoader(public_train_dest, shuffle=False, batch_size=args.edge_bs, num_workers=0, collate_fn=public_train_dest.custom_collate_fn)

    with torch.no_grad():
        for step, (color, depth, inertial, skeleton, target,prompt_template,trainFlag,selected_idxs) in enumerate(public_train_dataloader):
            inputData = [torch.unsqueeze(depth, 1).cuda().float(), inertial.permute(0, 2, 1).cuda().float(),skeleton.cuda().float()]
            target = target.cuda().long()

            fused_feat,text_feat,logits= proxy(inputData,trainFlag,prompt_template,device,return_components=True)



            last_peft_output = get_last_peft_output(model=proxy, detach=True, return_global=return_global,return_personal=return_personal,return_w=return_w) if return_feat else None

            for idx in range(target.shape[0]):
                selected_idx = selected_idxs[idx]
                if last_peft_output is not None:
                    last_peft_output={key: last_peft_output[key] if last_peft_output[key] is not None else None for key in last_peft_output.keys()}

                local_eval_result[selected_idx] = {
                    'local_output': logits[idx].detach().cpu() if return_logits else None,
                    'last_peft_output': last_peft_output,
                    'fusion_feat':fused_feat[idx] if return_feat else None,
                    'text_feat':text_feat if return_feat else None
                }


    save_path=os.path.join(check_path,'distil_cloud_train_data',f'{communication}_train.pt')

    torch.save({'data':local_eval_result,'trainFlag':edge_trainFlag_str}, save_path)

def update_cloud_train_data2edge(args,proxy,train_input_file,client_idx,check_path,communication,device,return_logits=False,return_feat=True,return_global=False,return_personal=False,return_w=False):
    public_train_path=train_input_file.replace(f'Client{client_idx}','Cloud') if client_idx is not None else train_input_file

    node_data = torch.load(public_train_path)
    data = node_data['data']
    prompt_template = node_data['prompt_template']
    modality_data={}
    for modality_name in node_data['data'].keys():
        trainFlag=np.array(modality_name.split('-'), dtype=np.float32)
        public_train_dest =SingleModalDataset(node_data['data'][modality_name],trainFlag,prompt_template)
        local_eval_result=[dict() for _ in range(public_train_dest.__len__())]
        proxy.eval()

        public_train_dataloader = DataLoader(public_train_dest, shuffle=False, batch_size=args.edge_bs, num_workers=0, collate_fn=public_train_dest.custom_collate_fn)

        with torch.no_grad():
            for step, (color, depth, inertial, skeleton, target,prompt_template,trainFlag,selected_idxs) in enumerate(public_train_dataloader):
                inputData = [torch.unsqueeze(depth, 1).cuda().float(), inertial.permute(0, 2, 1).cuda().float(),skeleton.cuda().float()]
                target = target.cuda().long()

                fused_feat,text_feat,logits= proxy(inputData,trainFlag,prompt_template,device,return_components=True)


                last_peft_output = get_last_peft_output(model=proxy, detach=True, return_global=return_global,return_personal=return_personal,return_w=return_w) if return_feat else None

                for idx in range(target.shape[0]):
                    selected_idx = selected_idxs[idx]
                    if last_peft_output is not None:
                        last_peft_output={key: last_peft_output[key] if last_peft_output[key] is not None else None for key in last_peft_output.keys()}

                    local_eval_result[selected_idx] = {
                        'local_output': logits[idx].detach().cpu() if return_logits else None,
                        'last_peft_output': last_peft_output,
                        'fusion_feat':fused_feat[idx] if return_feat else None,
                        'text_feat':text_feat if return_feat else None
                    }

        modality_data[modality_name]={
            'data':local_eval_result,
            'trainFlag':modality_name
        }
    save_path=os.path.join(check_path,'distil_cloud_train_data',f'{communication}_train.pt')

    torch.save(modality_data, save_path)




def write_to_excel(df,xlsx_name,check_path,communication):
    df = pd.concat(df, axis=1)
    df.columns = df.columns.get_level_values(-1)
    df['communication'] = communication
    save_path = os.path.join(check_path,'Test', xlsx_name)
    if os.path.exists(save_path):
        old_data = pd.read_excel(save_path)
        df = pd.concat([old_data, df], axis=0,ignore_index=True)
    df.to_excel(save_path, index=False)
    # print(f'################# sucessful save at {save_path} ########################')

def convert_list_to_str(x):
    if not isinstance(x,list):
        if isinstance(x,torch.Tensor):
            x=x.detach().cpu().to(torch.float32)

        x=list(np.array(x))
    x=[str(x[i]) for i in range(len(x))]
    return '，'.join(x)

def get_last_peft_output(model,detach=True,return_global=False,return_personal=False,return_w=False):
    this_module_output=None
    if isinstance(model.last_peft_layer, model.perTuckerLayer) or isinstance(model.last_peft_layer, model.SperTuckerLayer) or isinstance(model.last_peft_layer,model.VperTuckerLayer):
        this_module_output = {
            'global': model.last_peft_layer._g_x.detach().cpu() if return_global else None,
            'personal': model.last_peft_layer._p_x.detach().cpu() if return_personal else None,
            'w_output': model.last_peft_layer._w_x.detach().cpu() if return_w else None
        } if detach else {
            'global': model.last_peft_layer._g_x if return_global else None,
            'personal': model.last_peft_layer._p_x if return_personal else None,
            'w_output': model.last_peft_layer._w_x if return_w else None
        }

    elif isinstance(model.last_peft_layer, model.TuckerLayer) or isinstance(model.last_peft_layer, model.LoraLayer):
        this_module_output = {
            'global': model.last_peft_layer._g_x.detach().cpu() if return_global else None,
            'personal': model.last_peft_layer._g_x.detach().cpu() if return_personal else None,
            'w_output': model.last_peft_layer._w_x.detach().cpu() if return_w else None
        } if detach else {
            'global': model.last_peft_layer._g_x if return_global else None,
            'personal': model.last_peft_layer._g_x if return_personal else None,
            'w_output': model.last_peft_layer._w_x if return_w else None
        }
    return this_module_output


