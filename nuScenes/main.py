#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
import json
import sys
import ntpath
import math
import subprocess
from options import args_parser
from utils import setup_seed
from generate_distributed_data import data_processor
def check_exist_client_info_json(args,node_name):
    save_root_path, base_method_name = check_dir_name()
    check_path = os.path.join(save_root_path, os.path.join(base_method_name, node_name))
    if not os.path.exists(os.path.join(check_path, 'checkpoint')):
        print(f'预训练好的模型不存在：{os.path.join(check_path, "checkpoint")}')
        return 1,check_path
    else:
        model_num=len(os.listdir(os.path.join(check_path, 'checkpoint')))
        if model_num<args.communications:
            print(f'预训练好的模型数量存在缺失：{os.path.join(check_path, "checkpoint")}')
            return 1,check_path
        else:
            return 0,check_path

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
def check_dir_name():
    save_dir_path = os.path.join(os.path.dirname(sys.argv[0]), args.result_dir_name)
    data_info=f'C-{args.communications}_PR-{args.participate_data_ratio}_IID-{args.iid}_{args.data_type}'
    save_dir_path = os.path.join(save_dir_path, data_info)


    edge_backbone_name=[ntpath.basename(args.edge_model_path[client_idx]) for client_idx in range(args.client_num)]
    edge_backbone_name='-'.join(edge_backbone_name)
    backbone_name='{}-{}'.format(ntpath.basename(args.cloud_model_path),edge_backbone_name)


    para_name = f'rank{args.rank}-LoAlpha{args.lora_alpha}-lr{args.d2cloud_lr}' if args.aggregation == 'centralized' else f'rank{args.rank}-Cep{args.cloud_epochs}-dClr{args.d2cloud_lr}-dElr{args.d2edge_lr}-Elr{str(args.edge_lr)}'
    method_name = f'{args.aggregation}-E_{args.cloud_peft}' if args.aggregation == 'centralized' else f'{args.aggregation}-E_{args.peft}-C_{args.cloud_peft}'
    save_root_path = os.path.join(save_dir_path, '{}_{}'.format(backbone_name.replace('-Instruct',''), para_name)) if args.modal_iid==1 else os.path.join(save_dir_path, 'ModalHetero_{}_{}'.format(backbone_name.replace('-Instruct',''), para_name))
    return save_root_path, method_name





def build_cmd(args,client_index=None,edge_lr=0,script_name='local_update_for_one_edge.py',prev_edge_w_path=None,train_input_file=None,check_path=None,communication=0,root_path=None,only_test=None,generate_teacher=None):
    cmd=['deepspeed', '--num_gpus', str(args.gpus), script_name]
    fixed_args_name=['epochs','peft','aggregation','gcs','rank','lora_alpha','data_path','gpus','lambda_lm','seed_num','client_num','raw_data_path','communications']
    dynamic_args_name = ['edge_model_path','edge_bs']
    for name in fixed_args_name:
        cmd.append(f'--{name}')
        cmd.append(str(getattr(args,name)))

    if prev_edge_w_path is not None:
        cmd.append(f'--prev_edge_w_path')
        cmd.append(prev_edge_w_path)

    if client_index is not None:
        cmd.append(f'--client_idx')
        cmd.append(str(client_index))
        cmd.append(f'--peft')
        cmd.append(args.peft)
        for name in dynamic_args_name:
            cmd.append(f'--{name}')
            cmd.append(str(getattr(args, name)[client_index]))
    else:
        cmd.append(f'--edge_model_path')
        cmd.append(args.cloud_model_path)

        cmd.append(f'--edge_bs')
        cmd.append(str(args.edge_bs[-1]))

        cmd.append(f'--peft')
        cmd.append(args.cloud_peft)

    cmd.append(f'--edge_lr')
    cmd.append(str(edge_lr))

    if train_input_file is not None:
        cmd.append(f'--train_input_file')
        cmd.append(train_input_file)
    cmd.append(f'--check_path')
    cmd.append(check_path)
    cmd.append(f'--root_path')
    cmd.append(root_path)
    cmd.append(f'--communication')
    cmd.append(str(communication))
    if only_test is not None:
        cmd.append(f'--only_test')
        cmd.append(str(only_test))
    if generate_teacher is not None:
        cmd.append(f'--generate_teacher')
        cmd.append(str(generate_teacher))

    return cmd

def build_d2cloud_cmd(args,d2cloud_lr,script_name='distill2cloud.py',prev_cloud_w_path=None,train_input_file=None,check_path=None,communication=0,use_module=True):
    cmd=['deepspeed', '--num_gpus', str(args.gpus), script_name]
    fixed_args_name=['epochs','cloud_peft','gcs','rank','lora_alpha','data_path','gpus','cloud_epochs','client_num','cloud_model_path','ib_w','vib_w','discriminator_w','struct_w','communications','d2cloud_alpha','seed_num','aggregation','raw_data_path','lambda_lm']
    for name in fixed_args_name:
        cmd.append(f'--{name}')
        cmd.append(str(getattr(args,name)))
    cmd.append(f'--edge_bs')
    cmd.append(str(args.edge_bs[-1]))
    if prev_cloud_w_path is not None:
        cmd.append(f'--prev_cloud_w_path')
        cmd.append(prev_cloud_w_path)
    if train_input_file is not None:
        cmd.append(f'--train_input_file')
        cmd.append(train_input_file)
    cmd.append(f'--check_path')
    cmd.append(check_path)
    cmd.append(f'--communication')
    cmd.append(str(communication))
    cmd.append(f'--d2cloud_lr')
    cmd.append(str(d2cloud_lr))
    cmd.append(f'--use_module')
    cmd.append(str(use_module))
    return cmd

def build_d2edge_cmd(args,client_idx,d2edge_lr,script_name='distill2edge.py',prev_edge_w_path=None,train_input_file=None,check_path=None,communication=0,use_module=True):
    cmd=['deepspeed', '--num_gpus', str(args.gpus), script_name]
    fixed_args_name=['epochs','cloud_peft','gcs','rank','lora_alpha','data_path','gpus','logit_w','ce_w','T','struct_w','peft','edge_alpha','seed_num','communications','cloud_epochs','aggregation']
    for name in fixed_args_name:
        cmd.append(f'--{name}')
        cmd.append(str(getattr(args,name)))
    cmd.append(f'--edge_bs')
    cmd.append(str(args.edge_bs[client_idx]))
    cmd.append(f'--edge_model_path')
    cmd.append(str(args.edge_model_path[client_idx]))
    if prev_edge_w_path is not None:
        cmd.append(f'--prev_edge_w_path')
        cmd.append(prev_edge_w_path)
    cmd.append(f'--train_input_file')
    cmd.append(train_input_file)
    cmd.append(f'--check_path')
    cmd.append(check_path)
    cmd.append(f'--communication')
    cmd.append(str(communication))
    cmd.append(f'--client_idx')
    cmd.append(str(client_idx))
    cmd.append(f'--d2edge_lr')
    cmd.append(str(d2edge_lr))
    cmd.append(f'--use_module')
    cmd.append(str(use_module))
    return cmd


def cosine_annealing(communication, total_communication, initial_lr,min_lr=1e-5):
    cosine_value=math.cos(math.pi*communication/total_communication)
    return min_lr+0.5*(initial_lr-min_lr)*(1+cosine_value)

def run_subprocess(cmd,present_screen=False):
    try:
        completed_process =subprocess.run(cmd, check=True) if present_screen else subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error:", e)
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)





if __name__ == '__main__':
    args, argsDict = args_parser()
    args.dev = 'cuda'
    warnings.filterwarnings("ignore")
    setup_seed()

    root_path = os.path.join(r'./data/split_data', 'distributed',f'dataRatio-{args.participate_data_ratio}_ClientNum-{args.client_num}_IID-{args.iid}',args.data_type) if args.data_type.lower().strip() == 'full' \
        else os.path.join(r'./data/split_data', 'distributed',f'dataRatio-{args.participate_data_ratio}_ClientNum-{args.client_num}_IID-{args.iid}',f'{args.data_type}-{args.communications}')
    if not os.path.exists(root_path):
        data_op = data_processor(args.data_json_path)
        data_op.distributed_data_processor(root_path, args.participate_data_ratio, args.client_num, args.iid,args.data_type, args.communications)
    edge_check_path = {f'Client{client_idx}': None for client_idx in range(args.client_num)}
    start_communication = 0
    end_communication = args.communications


    edge_check_path['Cloud'] = generated_client_info_json(args,node_name='Cloud',start_communication=start_communication)
    exp_dirname = os.path.dirname(edge_check_path['Cloud'])
    for client_idx in range(args.client_num):
        edge_check_path[f'Client{client_idx}'] = generated_client_info_json(args,node_name=f'Client{client_idx}',start_communication=start_communication)



    for communication in range(args.startC,end_communication):
        for client_idx in range(args.client_num):
            if communication>0:
                print(f'#################### {args.aggregation}: start {communication-1}-th round D2Edge aggregation ####################')
                train_input_file = os.path.join(root_path, 'Cloud','train.json') if args.data_type.lower().strip() == 'full' \
                    else os.path.join(root_path, f'Cloud', f'{communication-1}_train.json')
                check_path = edge_check_path[f'Client{client_idx}']
                prev_edge_w_path = os.path.join(edge_check_path[f'Client{client_idx}'], 'checkpoint',f'local_w_C{communication-1}.pth') if communication > 0 else None

                use_module=True
                cmd = build_d2edge_cmd(args=args, client_idx=client_idx, d2edge_lr=args.d2edge_lr, script_name='distill2edge.py',prev_edge_w_path=prev_edge_w_path, train_input_file=train_input_file,check_path=check_path, communication=communication,use_module=use_module)
                run_subprocess(cmd,present_screen=False)


            print(f'#################### {args.aggregation}: start training {communication}-th communication || Client{client_idx} ###################')
            train_input_file = os.path.join(root_path, f'Client{client_idx}','train.json') if args.data_type.lower().strip() == 'full' \
                else os.path.join(root_path, f'Client{client_idx}', f'{communication}_train.json')
            check_path = edge_check_path[f'Client{client_idx}']
            prev_edge_w_path = os.path.join(edge_check_path[f'Client{client_idx}'], 'checkpoint',f'received_w_C{communication}.pth') if communication > 0 else None
            cmd=build_cmd(args=args,client_index=client_idx,edge_lr=cosine_annealing(communication+1, args.communications, args.edge_lr[client_idx],min_lr=1e-5),script_name='local_update_for_one_edge.py',prev_edge_w_path=prev_edge_w_path,train_input_file=train_input_file,check_path=check_path,communication=communication,root_path=root_path)
            run_subprocess(cmd,present_screen=False)


        print(f'#################### {args.aggregation}: start {communication}-th round D2Cloud aggregation ####################')
        train_input_file = os.path.join(root_path, 'Cloud','train.json') if args.data_type.lower().strip() == 'full' \
            else os.path.join(root_path, f'Cloud', f'{communication}_train.json')
        check_path = edge_check_path['Cloud']
        prev_cloud_w_path=os.path.join(edge_check_path['Cloud'],'checkpoint',f'cloud_w_C{communication-1}.pth') if communication>0 else None

        use_module = args.use_module[1] if args.run_type == 'ablation' else True
        cmd = build_d2cloud_cmd(args=args, d2cloud_lr=args.d2cloud_lr,script_name='distill2cloud.py',prev_cloud_w_path=prev_cloud_w_path, train_input_file=train_input_file,check_path=check_path, communication=communication,use_module=use_module)
        run_subprocess(cmd,present_screen=False)



