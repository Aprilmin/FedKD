import os
from pathlib import Path
import pandas as pd
import datetime as dt
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import deepspeed.comm as dist
from torch.utils.data import DistributedSampler,DataLoader
from nuscenes_dataset import MultiFrameDataset4Inference
from generateTeacherData import integrate_curvature_for_points,compute_speed_curv_metrics,compute_traj_ADE


def test_personal_generalize_metrics(args,proxy,train_input_file,check_path,communication,node_name,batch_size):
    test_input_file_zoos={
        client_name:os.path.join(Path(train_input_file).parent.parent,client_name,'test.json') for client_name in os.listdir(Path(train_input_file).parent.parent)
    }

    test_result_saved_root_zoos={
        client_name:os.path.join(check_path,'Test',client_name) for client_name in test_input_file_zoos.keys()
    }
    test_result_zoos={}
    for test_data_owned_name in test_input_file_zoos.keys():
        if dist.get_rank() == 0:
            if not os.path.exists(test_result_saved_root_zoos[test_data_owned_name]):
                os.makedirs(test_result_saved_root_zoos[test_data_owned_name])
        is_plot =  False
        test_trained_model(args,proxy,node_name=node_name,
                                       test_data_path=test_input_file_zoos[test_data_owned_name],
                                       test_result_saved_root=test_result_saved_root_zoos[test_data_owned_name],
                                       communication=communication,is_plot=is_plot,batch_size=batch_size)

    # compute generalized/personal metrics
    if dist.get_rank() == 0:
        for test_data_owned_name in test_input_file_zoos.keys():
            test_result_zoos[test_data_owned_name]=torch.load(os.path.join(test_result_saved_root_zoos[test_data_owned_name],'merged_result.pt'), weights_only=False)


        computed_metrics={}
        computed_metrics['generalized']={metric:[] for metric in test_result_zoos['Cloud'].keys()}
        computed_metrics['personal'] = {test_data_owned_name:test_result_zoos[test_data_owned_name] for test_data_owned_name in test_result_zoos.keys()}

        for metric in computed_metrics['generalized'].keys():
            for client_name in test_result_zoos.keys():
                if isinstance(test_result_zoos[client_name][metric],torch.Tensor):
                    test_result_zoos[client_name][metric]=test_result_zoos[client_name][metric].cpu().numpy()
                computed_metrics['generalized'][metric].append(test_result_zoos[client_name][metric])


            computed_metrics['generalized'][metric]=np.mean(computed_metrics['generalized'][metric],axis=0)
            if isinstance(computed_metrics['generalized'][metric],dt.timedelta):
                computed_metrics['generalized'][metric] = pd.DataFrame(data=[computed_metrics['generalized'][metric]],columns=[metric])

                for tmp_name in computed_metrics['personal'].keys():
                    computed_metrics['personal'][tmp_name][metric]=pd.DataFrame(data=[computed_metrics['personal'][tmp_name][metric]],columns=[metric])

            else:
                col_name=[f'{metric}_{idx}s' for idx in range(len(computed_metrics['generalized'][metric]))]
                computed_metrics['generalized'][metric]=pd.DataFrame(data=[computed_metrics['generalized'][metric]],columns=col_name)

                for tmp_name in computed_metrics['personal'].keys():
                    computed_metrics['personal'][tmp_name][metric]=pd.DataFrame(data=[computed_metrics['personal'][tmp_name][metric]],columns=col_name)
        for tmp_name in computed_metrics['personal'].keys():
            if ('client' in node_name.lower() and node_name==tmp_name) or ('client' not in node_name.lower() and 'cloud' not in tmp_name.lower()):
                write_to_excel(df=computed_metrics['personal'][tmp_name], xlsx_name=f'{tmp_name}.xlsx', check_path=check_path,communication=communication)



def test_trained_model(args,proxy,test_data_path,test_result_saved_root,communication=None,is_plot=False,batch_size=1,node_name=None):

    speed_std = 1.8046681286146409
    speed_mean = 2.5373214399490234
    curv_std = 0.07511833558384672
    curv_mean = 0.021407328091665685
    speed_min = 0
    speed_max = 10.082848494377343

    test_dest = MultiFrameDataset4Inference(raw_sample_root=args.raw_data_path,sample_root=args.data_path, input_file=test_data_path,processor=proxy.processor, model_name=proxy.model_name,is_test=True)
    local_eval_result = [list() for _ in range(test_dest.__len__())]
    proxy.eval()
    sampler = DistributedSampler(test_dest, num_replicas=args.gpus, rank=torch.distributed.get_rank())
    test_dataloader = DataLoader(test_dest, shuffle=False, batch_size=batch_size, num_workers=0, sampler=sampler,collate_fn=test_dest.custom_collate_fn)
    sampler.set_epoch(0)
    progress_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))


    pic_save_dir=os.path.join(test_result_saved_root,'Plot')


    with torch.no_grad():
        for step, (inputs, labels, selected_idxs, future_speed_and_curvatur, future_trajs,fut_start_world,obs_ego_velocities,obs_camera_params,obs_ego_poses,image_paths,plot_imgs_paths) in progress_bar:

            start_time = datetime.now()
            inputs = {
                k: v.to(proxy.device) for k, v in inputs.items()
            }

            _, speed_curvatures_pred,_ = proxy(inputs, labels)
            inference_time=datetime.now()-start_time
            future_speed_and_curvatur = torch.tensor(future_speed_and_curvatur, device=proxy.device,dtype=torch.float32)


            speed_curvatures_pred[:, :, 0] = speed_curvatures_pred[:, :, 0] * (speed_max - speed_min) + speed_min
            speed_curvatures_pred[:, :, 1] = speed_curvatures_pred[:, :, 1] * curv_std + curv_mean

            future_speed_and_curvatur[:, :, 0] = future_speed_and_curvatur[:, :, 0] * (speed_max - speed_min) + speed_min
            future_speed_and_curvatur[:, :, 1] = future_speed_and_curvatur[:, :, 1] * curv_std + curv_mean

            computed_pred_trajs = integrate_curvature_for_points(curvatures=speed_curvatures_pred[:, :, 1],
                                                        speeds=speed_curvatures_pred[:, :, 0],
                                                        init_pos=torch.tensor([x[:2] for x in fut_start_world], dtype=torch.float32,device=proxy.device),
                                                        init_heading=torch.atan2(
                                                            torch.tensor([v[-1][1] for v in obs_ego_velocities]),
                                                            torch.tensor([v[-1][0] for v in obs_ego_velocities])
                                                        ).to(proxy.device))
            computed_real_trajs=integrate_curvature_for_points(curvatures=future_speed_and_curvatur[:, :, 1],
                                                        speeds=future_speed_and_curvatur[:, :, 0],
                                                        init_pos=torch.tensor([x[:2] for x in fut_start_world], dtype=torch.float32,device=proxy.device),
                                                        init_heading=torch.atan2(
                                                            torch.tensor([v[-1][1] for v in obs_ego_velocities]),
                                                            torch.tensor([v[-1][0] for v in obs_ego_velocities])
                                                        ).to(proxy.device))

            for idx in range(speed_curvatures_pred.shape[0]):
                selected_idx = selected_idxs[idx]
                this_speed_curvatures_pred=speed_curvatures_pred[idx]
                this_future_speed_and_curvatur=future_speed_and_curvatur[idx]
                this_obs_camera_params=obs_camera_params[idx]
                this_obs_ego_poses=obs_ego_poses[idx]
                this_image_paths=plot_imgs_paths[idx]

                speed_mse_sec_list,curv_mse_sec_list,speed_rmse_sec_list,curv_rmse_sec_list=compute_speed_curv_metrics(this_speed_curvatures_pred,this_future_speed_and_curvatur)
                traj_ade_sec_list=compute_traj_ADE(computed_pred_trajs[idx].detach().cpu(),computed_real_trajs[idx].detach().cpu())

                local_eval_result[selected_idx]={
                    'speed_mse':speed_mse_sec_list,
                    'curv_mse':curv_mse_sec_list,
                    'speed_rmse':speed_rmse_sec_list,
                    'curv_rmse':curv_rmse_sec_list,
                    'traj_ade':traj_ade_sec_list,
                    'inference_time':inference_time,

                    'pred_speed':this_speed_curvatures_pred[:,0].detach().cpu(),
                    'real_speed':this_future_speed_and_curvatur[:,0].detach().cpu(),
                    'pred_curv':this_speed_curvatures_pred[:,1].detach().cpu(),
                    'real_curv':this_future_speed_and_curvatur[:,1].detach().cpu(),
                    'pred_traj':computed_pred_trajs[idx].detach().cpu(),
                    'real_traj':computed_real_trajs[idx].detach().cpu(),



                }


    dist.barrier()
    all_results = [None for _ in range(dist.get_world_size())]

    torch.distributed.all_gather_object(all_results, local_eval_result)

    if dist.get_rank() == 0:
        merged_result={key:[[] for _ in range(len(local_eval_result))] for key in ['speed_mse','curv_mse','speed_rmse','curv_rmse','traj_ade','inference_time','pred_speed','real_speed','pred_curv','real_curv','pred_traj','real_traj']}
        for result in all_results:
            for i, entry in enumerate(result):
                if entry:  # 只覆盖非空的 entry
                    for key in merged_result.keys():
                        merged_result[key][i]=entry[key]

        for key in ['speed_mse','curv_mse','speed_rmse','curv_rmse','traj_ade','inference_time']:
            merged_result[key]=np.mean(merged_result[key],axis=0)
            if isinstance(merged_result[key],torch.Tensor):
                merged_result[key]=merged_result[key].detach().cpu().to(torch.float32)



        merged_result={metric:merged_result[metric] for metric in ['speed_mse','curv_mse','speed_rmse','curv_rmse','traj_ade','inference_time']}
        torch.save(merged_result,os.path.join(test_result_saved_root,'merged_result.pt'))





def update_cloud_train_data(args,proxy,train_input_file,client_idx,check_path,communication,return_logits=False,return_feat=True,return_global=False,return_personal=False,return_w=False):
    public_train_path=train_input_file.replace(f'Client{client_idx}','Cloud') if client_idx is not None else train_input_file
    public_train_dest = MultiFrameDataset4Inference(raw_sample_root=args.raw_data_path,sample_root=args.data_path, input_file=public_train_path, processor=proxy.processor,model_name=proxy.model_name)
    local_eval_result=[dict() for _ in range(public_train_dest.__len__())]
    proxy.eval()

    speed_min =0
    speed_max = 10.082848494377343
    curv_std = 0.07511833558384672
    curv_mean = 0.021407328091665685

    sampler = DistributedSampler(public_train_dest, num_replicas=args.gpus, rank=torch.distributed.get_rank())
    update_bz=1
    public_train_dataloader = DataLoader(public_train_dest, shuffle=False, batch_size=update_bz, num_workers=0, sampler=sampler,collate_fn=public_train_dest.custom_collate_fn)
    sampler.set_epoch(0)
    progress_bar = tqdm(enumerate(public_train_dataloader), total=len(public_train_dataloader))
    with torch.no_grad():
        for step, (inputs, labels, selected_idxs,future_speed_and_curvatur) in progress_bar:

            inputs = {
                k: v.to(proxy.device) for k, v in inputs.items()
            }
            # labels = labels.to(proxy.device)
            future_speed_and_curvatur = torch.tensor(future_speed_and_curvatur, device=proxy.device,dtype=torch.float32)
            generate_text, head_outputs,pooled_features = proxy(inputs,labels)
            head_outputs[:, :, 0] = head_outputs[:, :, 0] * (speed_max-speed_min) + speed_min
            head_outputs[:, :, 1] = head_outputs[:, :, 1] * curv_std + curv_mean

            future_speed_and_curvatur[:, :, 0] = future_speed_and_curvatur[:, :, 0] * (speed_max-speed_min) + speed_min
            future_speed_and_curvatur[:, :, 1] = future_speed_and_curvatur[:, :, 1] * curv_std + curv_mean
            res_logits=head_outputs-future_speed_and_curvatur

            last_peft_output = get_last_peft_output(model=proxy, detach=True, return_global=return_global,return_personal=return_personal,return_w=return_w) if return_feat else None
            for idx in range(update_bz):
                selected_idx = selected_idxs[idx]
                if last_peft_output is not None:
                    last_peft_output={key: last_peft_output[key][idx] if last_peft_output[key] is not None else None for key in last_peft_output.keys()}
                    last_peft_output['pooled_features']=pooled_features[idx].detach().cpu()
                local_eval_result[selected_idx] = {
                    'local_output': res_logits[idx].detach().cpu() if return_logits is not None else None,
                    'last_peft_output': last_peft_output
                }

    all_results = [None for _ in range(dist.get_world_size())]
    torch.distributed.all_gather_object(all_results, local_eval_result)
    save_path=os.path.join(check_path,'distil_cloud_train_data',f'{communication}_train.pt')
    if dist.get_rank() == 0:
        merged_result = [{} for _ in range(len(local_eval_result))]
        for result in all_results:
            for i, entry in enumerate(result):
                if entry:  # 只覆盖非空的 entry
                    merged_result[i] = entry
        torch.save(merged_result, save_path)



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
    if isinstance(model.last_peft_layer, model.perTuckerLayer) :
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
            'w_output': model.last_peft_layer._w_x.detach().cpu() if return_w else None
        } if detach else {
            'global': model.last_peft_layer._g_x if return_global else None,
            'w_output': model.last_peft_layer._w_x if return_w else None
        }
    return this_module_output


