from tqdm import tqdm
from lora_model import DriveMLLM,Edge2CloudDistil
from nuscenes_dataset import MultiFrameDataset4Inference
import argparse
import os
import numpy as np
import copy
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
import torch
from datetime import datetime

def create_proxy(args,client_idx,load_model_path,updated_w_path=None):
    proxy = DriveMLLM(rank=args.rank, lora_alpha=args.lora_alpha, peft=args.peft.lower(),pretrained_model_path=load_model_path,is_test=True) if client_idx is not None \
        else Edge2CloudDistil(rank=args.rank, lora_alpha=args.lora_alpha, peft=args.cloud_peft.lower(),pretrained_model_path=load_model_path,client_num=args.client_num,lambda_grl = 1.0,is_test=True)
    if updated_w_path is not None:
        this_prev_edge_w=torch.load(updated_w_path, weights_only=False)['model']
        old_w=copy.deepcopy(proxy.state_dict())
        proxy.load_state_dict(this_prev_edge_w, strict=False)
        new_w=copy.deepcopy(proxy.state_dict())
        distance_old=0
        distance_new=0


        for name in this_prev_edge_w.keys():
            distance_old+=torch.sum(old_w[name]-this_prev_edge_w[name])
            distance_new+=torch.sum(new_w[name]-this_prev_edge_w[name])
        if distance_old != 0 and distance_new != 0:
            print(
                f'##########################{distance_old}, {distance_new} fail to load model: {updated_w_path} ###########################')
            return None
        else:
            return proxy
    else:
        return proxy


def update_cloud_data(dev,args,client_idx,load_model_path,updated_w_path,train_input_file,check_path,communication,return_logits=False,return_feat=True,return_global=False,return_personal=False,return_w=False):
    proxy=create_proxy(args,client_idx,load_model_path,updated_w_path)

    if proxy is not None:

        public_train_path=train_input_file.replace(f'Client{client_idx}','Cloud') if client_idx is not None else train_input_file
        public_train_dest = MultiFrameDataset4Inference(sample_root=args.data_path, input_file=public_train_path, processor=proxy.processor,model_name=proxy.model_name)
        # proxy.to(dev)
        proxy.eval()

        eval_result = []

        with torch.no_grad():
            pbar=tqdm(range(public_train_dest.__len__()))

            for sample_idx in pbar:
                inputs, labels, selected_idx = public_train_dest.custom_collate_fn([public_train_dest.__getitem__(sample_idx)])

                output_texts_trimmed=proxy.generate(inputs) if return_logits else [None]

                if return_feat:
                        _,pred_speed_and_curvatur=proxy(inputs,labels)
                        last_peft_output = get_last_peft_output(model=proxy, detach=True, return_global=return_global,return_personal=return_personal,return_w=return_w)
                else:
                    last_peft_output=None

                eval_result.append({
                    'local_output': output_texts_trimmed[0],
                    'last_peft_output': {
                        key: last_peft_output[key][0] if last_peft_output[key] is not None else None for key in last_peft_output.keys()} if last_peft_output is not None else None
                })

        torch.save(eval_result,os.path.join(check_path,'distil_cloud_train_data',f'{communication}_train.pt'))
        assert os.path.exists(os.path.join(check_path,'distil_cloud_train_data',f'{communication}_train.pt'))
    return proxy





def eval_local_test_data(args,proxy,test_result_saved_root,test_data_path,communication,dev):
    speed_std = 1.8046681286146409
    speed_mean = 2.5373214399490234
    curv_std = 0.07511833558384672
    curv_mean = 0.021407328091665685

    test_dest = MultiFrameDataset4Inference(sample_root=args.data_path, input_file=test_data_path,processor=proxy.processor, model_name=proxy.model_name,is_test=True)

    prediction_file_path=os.path.join(test_result_saved_root,f'C{communication}.csv') if communication is not None else os.path.join(test_result_saved_root,f'ZeroShot.csv')
    pic_save_dir=os.path.join(test_result_saved_root,'Plot')
    if not os.path.exists(pic_save_dir):
        os.makedirs(pic_save_dir)

    with open(prediction_file_path, 'w', encoding='utf-8') as f:
        f.write('sampleIdx|llm_text|pred_speed|real_speed|pred_curv|real_curv|speed_mse|curv_mse|speed_rmse|curv_rmse|traj_ade|inference_times\n')
    proxy.eval()
    speed_mse_list,curv_mse_list,speed_rmse_list,curv_rmse_list,traj_ade_list,inference_time_list=[],[],[],[],[],[],[]
    with torch.no_grad():
        pbar=tqdm(range(len(test_dest.__len__())))
        for sample_idx in pbar:
            inputs, labels, selected_idx, future_speed_and_curvatur, future_trajs,fut_start_world,obs_ego_velocities,obs_camera_params,obs_ego_poses,image_paths=test_dest.custom_collate_fn([test_dest.__getitem__(sample_idx)])

            start_time = datetime.now()
            output_texts_trimmed = proxy.generate(inputs)
            text_time=datetime.now()-start_time
            start_time = datetime.now()
            _, speed_curvatures_pred = proxy(inputs, labels)
            value_time=datetime.now()-start_time
            inference_time=[text_time,value_time]


            speed_curvatures_pred[:, :, 0] = speed_curvatures_pred[:, :, 0] * speed_std + speed_mean
            speed_curvatures_pred[:, :, 1] = speed_curvatures_pred[:, :, 1] * curv_std + curv_mean

            future_speed_and_curvatur[:, :, 0] = future_speed_and_curvatur[:, :, 0] * speed_std + speed_mean
            future_speed_and_curvatur[:, :, 1] = future_speed_and_curvatur[:, :, 1] * curv_std + curv_mean

            computed_pred_trajs = integrate_curvature_for_points(curvatures=torch.tensor(speed_curvatures_pred[:, :, 1],dtype=torch.float32),
                                                        speeds=torch.tensor(speed_curvatures_pred[:, :, 0], dtype=torch.float32),
                                                        init_pos=torch.tensor([x[:2] for x in fut_start_world], dtype=torch.float32),
                                                        init_heading=torch.atan2(
                                                            torch.tensor([v[-1][1] for v in obs_ego_velocities]),
                                                            torch.tensor([v[-1][0] for v in obs_ego_velocities])
                                                        ))
            computed_real_trajs=integrate_curvature_for_points(curvatures=torch.tensor(future_speed_and_curvatur[:, :, 1],dtype=torch.float32),
                                                        speeds=torch.tensor(future_speed_and_curvatur[:, :, 0], dtype=torch.float32),
                                                        init_pos=torch.tensor([x[:2] for x in fut_start_world], dtype=torch.float32),
                                                        init_heading=torch.atan2(
                                                            torch.tensor([v[-1][1] for v in obs_ego_velocities]),
                                                            torch.tensor([v[-1][0] for v in obs_ego_velocities])
                                                        ))



            output_texts_trimmed=output_texts_trimmed[0]
            speed_curvatures_pred=speed_curvatures_pred[0]
            future_speed_and_curvatur=future_speed_and_curvatur[0]
            obs_camera_params=obs_camera_params[0]
            obs_ego_poses=obs_ego_poses[0]
            image_paths=image_paths[0]

            speed_mse_sec_list,curv_mse_sec_list,speed_rmse_sec_list,curv_rmse_sec_list=compute_speed_curv_metrics(speed_curvatures_pred,future_speed_and_curvatur)
            traj_ade_sec_list=compute_traj_ADE(computed_pred_trajs,computed_real_trajs)


            speed_mse_list.append(speed_mse_sec_list)
            curv_mse_list.append(curv_mse_sec_list)
            speed_rmse_list.append(speed_rmse_sec_list)
            curv_rmse_list.append(curv_rmse_sec_list)
            traj_ade_list.append(traj_ade_sec_list)
            inference_time_list.append(inference_time)
            with open(prediction_file_path, 'a', encoding='utf-8') as f:
                f.write(f'{sample_idx}|{output_texts_trimmed}|{str(speed_curvatures_pred[:,0]).replace("[","").replace("]","")}|{str(future_speed_and_curvatur[:,0]).replace("[","").replace("]","")}|{str(speed_curvatures_pred[:,1]).replace("[","").replace("]","")}|{str(future_speed_and_curvatur[:,1]).replace("[","").replace("]","")}'
                        f'|{str(speed_mse_sec_list).replace("[","").replace("]","")}|{str(curv_mse_sec_list).replace("[","").replace("]","")}|{str(speed_rmse_sec_list).replace("[","").replace("]","")}|{str(curv_rmse_sec_list).replace("[","").replace("]","")}|{str(traj_ade_sec_list).replace("[","").replace("]","")}|{str(inference_time).replace("[","").replace("]","")}\n')
        speed_mse_mean=np.mean(speed_mse_list,axis=0)
        curv_mse_mean=np.mean(curv_mse_list,axis=0)
        speed_rmse_mean=np.mean(speed_rmse_list,axis=0)
        curv_rmse_mean=np.mean(curv_rmse_list,axis=0)
        traj_ade_mean=np.mean(traj_ade_list,axis=0)
        inference_time_mean=np.mean(inference_time_list,axis=0)
        return speed_mse_mean,curv_mse_mean,speed_rmse_mean,curv_rmse_mean,traj_ade_mean,inference_time_mean


def compute_traj_ADE(pred_traj,fut_ego_traj_world):
    pred_len=pred_traj.shape[0]
    pred_traj = np.array(pred_traj)
    fut_ego_traj_world = np.array(fut_ego_traj_world)
    # ade = np.mean(np.linalg.norm(fut_ego_traj_world[:pred_len] - pred_traj, axis=1))
    pred1_len = min(pred_len, 2)
    ade1s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred1_len] - pred_traj[:pred1_len], axis=1))
    # ade1s_list.append(ade1s)

    pred2_len = min(pred_len, 4)
    ade2s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred2_len] - pred_traj[:pred2_len], axis=1))
    # ade2s_list.append(ade2s)

    pred3_len = min(pred_len, 6)
    ade3s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred3_len] - pred_traj[:pred3_len], axis=1))

    pred4_len = min(pred_len, 8)
    ade4s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred4_len] - pred_traj[:pred4_len], axis=1))

    pred5_len = min(pred_len, 10)
    ade5s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred5_len] - pred_traj[:pred5_len], axis=1))
    return [ade1s,ade2s,ade3s,ade4s,ade5s]

def compute_speed_curv_metrics(pred_speed_curv,real_speed_curv):
    pred_len=pred_speed_curv.shape[0]

    return [
        torch.nn.functional.mse_loss(pred_speed_curv[:min(pred_len, 2),0],real_speed_curv[:min(pred_len, 2),0]).item(),
        torch.nn.functional.mse_loss(pred_speed_curv[:min(pred_len, 4), 0], real_speed_curv[:min(pred_len, 4), 0]).item(),
        torch.nn.functional.mse_loss(pred_speed_curv[:min(pred_len, 6), 0], real_speed_curv[:min(pred_len, 6), 0]).item(),
        torch.nn.functional.mse_loss(pred_speed_curv[:min(pred_len, 8), 0], real_speed_curv[:min(pred_len, 8), 0]).item(),
        torch.nn.functional.mse_loss(pred_speed_curv[:min(pred_len, 10), 0], real_speed_curv[:min(pred_len, 10), 0]).item()
    ],[
        torch.nn.functional.mse_loss(pred_speed_curv[:min(pred_len, 2), 1], real_speed_curv[:min(pred_len, 2), 1]).item(),
        torch.nn.functional.mse_loss(pred_speed_curv[:min(pred_len, 4), 1], real_speed_curv[:min(pred_len, 4), 1]).item(),
        torch.nn.functional.mse_loss(pred_speed_curv[:min(pred_len, 6), 1], real_speed_curv[:min(pred_len, 6), 1]).item(),
        torch.nn.functional.mse_loss(pred_speed_curv[:min(pred_len, 8), 1], real_speed_curv[:min(pred_len, 8), 1]).item(),
        torch.nn.functional.mse_loss(pred_speed_curv[:min(pred_len, 10), 1], real_speed_curv[:min(pred_len, 10), 1]).item()
    ],[
        math.sqrt(torch.nn.functional.mse_loss(pred_speed_curv[:min(pred_len, 2), 0], real_speed_curv[:min(pred_len, 2), 0]).item()),
        math.sqrt(torch.nn.functional.mse_loss(pred_speed_curv[:min(pred_len, 4), 0], real_speed_curv[:min(pred_len, 4), 0]).item()),
        math.sqrt(torch.nn.functional.mse_loss(pred_speed_curv[:min(pred_len, 6), 0], real_speed_curv[:min(pred_len, 6), 0]).item()),
        math.sqrt(torch.nn.functional.mse_loss(pred_speed_curv[:min(pred_len, 8), 0], real_speed_curv[:min(pred_len, 8), 0]).item()),
        math.sqrt(torch.nn.functional.mse_loss(pred_speed_curv[:min(pred_len, 10), 0], real_speed_curv[:min(pred_len, 10), 0]).item())
    ],[
        math.sqrt(torch.nn.functional.mse_loss(pred_speed_curv[:min(pred_len, 2), 1], real_speed_curv[:min(pred_len, 2), 1]).item()),
        math.sqrt(torch.nn.functional.mse_loss(pred_speed_curv[:min(pred_len, 4), 1], real_speed_curv[:min(pred_len, 4), 1]).item()),
        math.sqrt(torch.nn.functional.mse_loss(pred_speed_curv[:min(pred_len, 6), 1], real_speed_curv[:min(pred_len, 6), 1]).item()),
        math.sqrt(torch.nn.functional.mse_loss(pred_speed_curv[:min(pred_len, 8), 1], real_speed_curv[:min(pred_len, 8), 1]).item()),
        math.sqrt(torch.nn.functional.mse_loss(pred_speed_curv[:min(pred_len, 10), 1], real_speed_curv[:min(pred_len, 10), 1]).item())
    ]


def integrate_curvature_for_points(curvatures, speeds, init_pos, init_heading, dt=1.0):
    """
    计算未来轨迹，使用梯形积分法（trapezoidal rule），支持 batch，可导。

    Args:
        curvatures: Tensor (B, T)          -> 每个时间步的曲率
        speeds: Tensor (B, T)              -> 每个时间步的速度
        init_pos: Tensor (B, 2)            -> 初始位置 [x, y]
        init_heading: Tensor (B,)          -> 初始航向角 θ0
        dt: float                          -> 时间间隔（假设每步为1秒）

    Returns:
        traj: Tensor (B, T, 2)             -> 每个时间步的 [x, y]
    """

    B, T = curvatures.shape

    # 累积曲率积分得到 θ
    delta_theta = curvatures * speeds * dt  # (B, T)
    theta = torch.cumsum(delta_theta, dim=1)  # (B, T)
    theta = theta + init_heading.unsqueeze(1)  # θ₀ 加到每一行

    # 计算速度方向上的分量
    vx = speeds * torch.cos(theta)  # (B, T)
    vy = speeds * torch.sin(theta)  # (B, T)

    # 梯形积分 vx 和 vy 得到 x 和 y 坐标
    x = torch.cumsum((vx[:, 1:] + vx[:, :-1]) / 2 * dt, dim=1)  # (B, T-1)
    y = torch.cumsum((vy[:, 1:] + vy[:, :-1]) / 2 * dt, dim=1)  # (B, T-1)

    # 插入起点
    x = torch.cat([torch.zeros(B, 1, device=x.device), x], dim=1)
    y = torch.cat([torch.zeros(B, 1, device=y.device), y], dim=1)

    x = x + init_pos[:, 0].unsqueeze(1)  # 加初始位置 x₀
    y = y + init_pos[:, 1].unsqueeze(1)  # 加初始位置 y₀

    return torch.stack([x, y], dim=2)  # (B, T, 2)


def get_last_peft_output(model,detach=True,return_global=False,return_personal=False,return_w=False):
    this_module_output=None
    if isinstance(model.last_peft_layer, model.perTuckerLayer) or isinstance(model.last_peft_layer, model.SperTuckerLayer):
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



    #
    # last_peft_layer = None
    #
    # for name, module in reversed(list(model.named_modules())):
    #     if isinstance(module, (model.perTuckerLayer, model.SperTuckerLayer, model.TuckerLayer, model.LoraLayer)):
    #         last_peft_layer = module
    #         print(f'&&&&&&&&&&&&&&&&&&&&&&& find the last_peft_layer: {name} &&&&&&&&&&&&&&&&&&&&&&&&&&')
    #         break
    # if last_peft_layer is None:
    #     raise ValueError("No PEFT layer found in the model.")
    # tensors_to_gather = []
    # if hasattr(last_peft_layer, "_g_x"):
    #     tensors_to_gather.append(last_peft_layer._g_x)
    # if hasattr(last_peft_layer, "_p_x"):
    #     tensors_to_gather.append(last_peft_layer._p_x)
    # if hasattr(last_peft_layer, "_w_x"):
    #     tensors_to_gather.append(last_peft_layer._w_x)
    #
    # with deepspeed.zero.GatheredParameters(tensors_to_gather, modifier_rank=None):
    #     result = {}
    #     if hasattr(last_peft_layer, "_g_x"):
    #         result['global'] = last_peft_layer._g_x.detach() if detach else last_peft_layer._g_x
    #     if hasattr(last_peft_layer, "_p_x"):
    #         result['personal'] = last_peft_layer._p_x.detach() if detach else last_peft_layer._p_x
    #     if hasattr(last_peft_layer, "_w_x"):
    #         result['w_output'] = last_peft_layer._w_x.detach() if detach else last_peft_layer._w_x
    # return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--cloud_peft', type=str, default='lora',choices=['lora', 'perTucker','Tlora','SperTucker'])
    parser.add_argument('--peft', type=str, default='lora', choices=['lora', 'perTucker', 'Tlora', 'SperTucker'])
    parser.add_argument('--data_path', type=str, default=r'/data/gm/DoubleCap/NuScenes/data')
    parser.add_argument('--client_num', type=int, default=2, help='number of users')

    parser.add_argument('--load_model_path', type=str, default=r'/data/gm/DoubleCap/models/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--updated_w_path', type=str, default=r'/data/gm/DoubleCap/NuScenes/src3_v2/result/distributed/C-20_PR-0.5_IID-0.1_nointersection/Qwen2.5-VL-7B-Instruct-Qwen2-VL-2B-Instruct-Qwen2.5-VL-3B-Instruct_rank64-LoAlpha32-seed42-Clr5e-05-Elr5e-05/distil-E_perTucker-C_lora/Cloud/checkpoint/cloud_w_C0.pth')
    parser.add_argument('--client_idx', type=int, default=None, help="cloud batch size: B")
    parser.add_argument('--train_input_file', type=str, default=r'/data/gm/DoubleCap/NuScenes/split_data/distributed/dataRatio-0.5_ClientNum-2_IID-0.1/nointersection-20/Cloud/0_train.json')
    parser.add_argument('--check_path', type=str,default=r'/data/gm/DoubleCap/NuScenes/src3_v2/result/distributed/C-20_PR-0.5_IID-0.1_nointersection/Qwen2.5-VL-7B-Instruct-Qwen2-VL-2B-Instruct-Qwen2.5-VL-3B-Instruct_rank64-LoAlpha32-seed42-Clr5e-05-Elr5e-05/distil-E_perTucker-C_lora/Cloud')
    parser.add_argument('--communication', type=int, default=None, help="cloud batch size: B")


    parser.add_argument('--return_logits', type=bool,default=True)
    parser.add_argument('--return_feat', type=bool, default=True)
    parser.add_argument('--return_global', type=bool, default=True)
    parser.add_argument('--return_personal', type=bool, default=False)
    parser.add_argument('--return_w', type=bool, default=True)


    parser.add_argument('--test_gate', type=bool, default=False)
    parser.add_argument('--train_gate', type=bool, default=True)




    args = parser.parse_args()

    dev = 'cuda'
    proxy=None
    if args.train_gate:
        proxy=update_cloud_data(dev,args,args.client_idx,args.load_model_path,args.updated_w_path,args.train_input_file,args.check_path,args.communication,args.return_logits,args.return_feat,args.return_global,args.return_personal,args.return_w)

    if args.test_gate:

        if proxy is None:
            proxy=create_proxy(args,args.client_idx,args.load_model_path,args.updated_w_path)

        if args.train_gate:
            test_data_path=os.path.join(os.path.dirname(args.train_input_file),'test.json')
            communication = args.communication
            test_result_saved_root = os.path.join(args.check_path, 'Test')
        else:

            test_data_path=args.test_data_path
            communication=args.communication if args.updated_w_path is not None else None
            test_result_saved_root = os.path.join(args.check_path, 'Test') if args.updated_w_path is not None else args.test_result_saved_root
        speed_mse_mean,curv_mse_mean,speed_rmse_mean,curv_rmse_mean,traj_ade_mean,inference_time_mean=eval_local_test_data(args, proxy, test_result_saved_root, test_data_path, communication, dev)
        if not os.path.exists(os.path.join(test_result_saved_root,'test_metrics.csv')):
            with open(os.path.join(test_result_saved_root,'test_metrics.csv'), 'w', encoding='utf-8') as f:
                f.write(f'communication|speed_mse|curv_mse|speed_rmse|curv_rmse|traj_ade|inference_time\n')
        with open(os.path.join(test_result_saved_root, 'test_metrics.csv'), 'w', encoding='utf-8') as f:
            f.write(f'{communication}|'
                    f'{str(speed_mse_mean).replace("[","").replace("]","")}|'
                    f'{str(curv_mse_mean).replace("[","").replace("]","")}|'
                    f'{str(speed_rmse_mean).replace("[","").replace("]","")}|'
                    f'{str(curv_rmse_mean).replace("[","").replace("]","")}|'
                    f'{str(traj_ade_mean).replace("[","").replace("]","")}|'
                    f'{str(inference_time_mean).replace("[","").replace("]","")}\n')


