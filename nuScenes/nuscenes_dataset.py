from torch.utils.data import Dataset
import os
import json
from qwen_vl_utils import process_vision_info
import torch
class MultiFrameDataset(Dataset):

    def __init__(self, sample_root,input_file, processor,dev=None,model_name=None):
        self.sample_root=sample_root
        with open(input_file,'r') as f:
            self.data = json.load(f)
        self.processor=processor
        self.tokenizer = processor.tokenizer
        self.dev=dev
        # self.model_name=model_name


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'img_path':[os.path.join(self.sample_root, self.data[idx]['current_imgs'][i]).replace('\\','/') for i in range(len(self.data[idx]['current_imgs']))],
            'obs_speed_and_curvatur_str':self.data[idx]['obs_speed_and_curvatur']['str'],
            'obs_speed_and_curvatur': self.data[idx]['obs_speed_and_curvatur']['float'],
            'future_speed_and_curvatur_str':self.data[idx]['future_speed_and_curvatur']['str'],
            'future_speed_and_curvatur': self.data[idx]['future_speed_and_curvatur']['float'],

            'obs_trajs_str': self.data[idx]['obs_trajs']['str'],
            'obs_trajs': self.data[idx]['obs_trajs']['float'],
            'future_trajs_str': self.data[idx]['future_trajs']['str'],
            'future_trajs': self.data[idx]['future_trajs']['float'],

            'selected_idx':idx,
            'fut_ego_traj_world':self.data[idx]['eval_meta']['fut_ego_traj_world'],
            'fut_start_world': self.data[idx]['eval_meta']['fut_start_world'],
            'obs_ego_velocities': self.data[idx]['eval_meta']['obs_ego_velocities']
        }


    def custom_collate_fn(self,features):

        selected_idx=[f['selected_idx'] for f in features]
        # obs_imgs,obs_speed_and_curvatur,future_speed_and_curvatur = zip(*batch)
        # image_paths = [f["img_path"] for f in features]
        # his_speed_and_curvatur_str = [f["his_speed_and_curvatur_str"] for f in features]
        # future_speed_and_curvatur_str = [f["future_speed_and_curvatur_str"] for f in features]
        # fut_ego_traj_world=[f["fut_ego_traj_world"] for f in features]
        # fut_start_world = [f["fut_start_world"] for f in features]
        # obs_ego_velocities = [f["obs_ego_velocities"] for f in features]
        future_speed_and_curvatur= [f["future_speed_and_curvatur"] for f in features]
        future_trajs=[f["future_trajs"] for f in features]


        prompts = []
        full_texts = []
        messages_list = []

        for idx in range(len(features)):
            prompt_text = (
                f"You are a autonomous driving assistant.You are provided with a sequence of 10 front-view camera images collected over the past 5 seconds. The corresponding speeds and curvatures of the ego vehicle at each time step are:{features[idx]['obs_speed_and_curvatur_str']}. Note: All speeds are normalized using min-max normalization and curvatures are normalized using Z-score normalization.\n "
                f"Your task is to analyze the driving history and the current frame (the last frame of the front-view camera images sequences) to predict future vehicle speeds and curvatures."
                f"Think aloud step by step:\n"
                f"(1) Scene understanding: Describe traffic lights, other vehicles, pedestrians, and lane markings.\n"
                f"(2) Risk awareness: Are there objects or agents the ego car should be cautious about?\n"
                f"(3) Driving intention: Based on the motion history and current view, is the car going straight, turning, slowing, or accelerating?\\n"
                
                f"Finally, predict the next 10 [speed, curvature] values of the ego car.\n"
                f"Format: [speed_1, curvature_1], [speed_2, curvature_2], ..., [speed_10, curvature_10]. Each value should be a float rounded to four decimal places."
            )
            label_text = features[idx]['future_speed_and_curvatur_str']
            # 构建消息体
            messages = [{
                "role": "user",
                "content": [
                    *[
                        {"type": "image", "image": f'file://{features[idx]["img_path"][i]}'} for i in range(len(features[idx]["img_path"]))
                    ],
                    {"type": "text", "text": prompt_text}
                ]
            }]
            messages_list.append(messages)
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            full_texts.append(prompt + label_text)

        image_inputs, video_inputs = process_vision_info(messages_list)

        # 使用 prompt + label_text 作为模型的输入
        inputs = self.processor(
            text=full_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        # 使用 tokenizer 构造 label，但把 prompt 部分 mask 掉，只保留 target 部分参与训练
        labels = inputs["input_ids"].clone()
        for i, prompt in enumerate(prompts):
            prompt_len = len(self.tokenizer(prompt).input_ids)
            labels[i, :prompt_len] = -100  # mask prompt 部分

        return inputs, labels, selected_idx,future_speed_and_curvatur,future_trajs

class MultiFrameDataset4Inference(Dataset):

    def __init__(self, raw_sample_root,sample_root, input_file, processor, dev=None, model_name=None,is_test=False):
        self.sample_root = sample_root
        with open(input_file, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.dev = dev
        # self.model_name=model_name
        self.is_test=is_test
        self.raw_sample_root=raw_sample_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'plot_img':os.path.join(self.raw_sample_root, self.data[idx]['current_imgs'][-1]).replace('\\','/'),
            'img_path':[os.path.join(self.sample_root, self.data[idx]['current_imgs'][i]).replace('\\','/') for i in range(len(self.data[idx]['current_imgs']))],
            'obs_speed_and_curvatur_str':self.data[idx]['obs_speed_and_curvatur']['str'],
            'obs_speed_and_curvatur': self.data[idx]['obs_speed_and_curvatur']['float'],
            'future_speed_and_curvatur_str':self.data[idx]['future_speed_and_curvatur']['str'],
            'future_speed_and_curvatur': self.data[idx]['future_speed_and_curvatur']['float'],

            'obs_trajs_str': self.data[idx]['obs_trajs']['str'],
            'obs_trajs': self.data[idx]['obs_trajs']['float'],
            'future_trajs_str': self.data[idx]['future_trajs']['str'],
            'future_trajs': self.data[idx]['future_trajs']['float'],

            'selected_idx':idx,
            'fut_ego_traj_world':self.data[idx]['eval_meta']['fut_ego_traj_world'],
            'fut_start_world': self.data[idx]['eval_meta']['fut_start_world'],
            'obs_ego_velocities': self.data[idx]['eval_meta']['obs_ego_velocities'],
            'obs_camera_params': self.data[idx]['eval_meta']['obs_camera_params'],
            'obs_ego_poses': self.data[idx]['eval_meta']['obs_ego_poses'],
        }

    def custom_collate_fn(self, features):
        selected_idx = [f['selected_idx'] for f in features]
        future_speed_and_curvatur = [f["future_speed_and_curvatur"] for f in features]
        prompts = []
        full_texts = []
        messages_list = []
        for idx in range(len(features)):
            prompt_text = (
                f"You are a autonomous driving assistant.You are provided with a sequence of 10 front-view camera images collected over the past 5 seconds. The corresponding speeds and curvatures of the ego vehicle at each time step are:{features[idx]['obs_speed_and_curvatur_str']}. Note: All speeds are normalized using min-max normalization and curvatures are normalized using Z-score normalization.\n "
                f"Your task is to analyze the driving history and the current frame (the last frame of the front-view camera images sequences) to predict future vehicle speeds and curvatures."
                f"Think aloud step by step:\n"
                f"(1) Scene understanding: Describe traffic lights, other vehicles, pedestrians, and lane markings.\n"
                f"(2) Risk awareness: Are there objects or agents the ego car should be cautious about?\n"
                f"(3) Driving intention: Based on the motion history and current view, is the car going straight, turning, slowing, or accelerating?\\n"

                f"Finally, predict the next 10 [speed, curvature] values of the ego car.\n"
                f"Format: [speed_1, curvature_1], [speed_2, curvature_2], ..., [speed_10, curvature_10]. Each value should be a float rounded to four decimal places."
            )

            # 构建消息体
            messages = [{
                "role": "user",
                "content": [
                    *[
                        {"type": "image", "image": f'file://{features[idx]["img_path"][i]}'} for i in
                        range(len(features[idx]["img_path"]))
                    ],
                    {"type": "text", "text": prompt_text}
                ]
            }]



            messages_list.append(messages)
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            full_texts.append(prompt)
        image_inputs, video_inputs = process_vision_info(messages_list)

        # 使用 prompt + label_text 作为模型的输入
        inputs = self.processor(
            text=full_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        labels = inputs["input_ids"].clone()
        for i, prompt in enumerate(prompts):
            prompt_len = len(self.tokenizer(prompt).input_ids)
            labels[i, :prompt_len] = -100  # mask prompt 部分

        if self.is_test:

            future_trajs = [f["future_trajs"] for f in features]
            fut_start_world=[f["fut_start_world"] for f in features]
            obs_ego_velocities=[f["obs_ego_velocities"] for f in features]

            obs_camera_params = [f["obs_camera_params"] for f in features]
            obs_ego_poses = [f["obs_ego_poses"] for f in features]

            image_paths = [f["img_path"] for f in features]
            plot_imgs_paths=[f["plot_img"] for f in features]
            return inputs, labels, selected_idx,future_speed_and_curvatur,future_trajs,fut_start_world,obs_ego_velocities,obs_camera_params,obs_ego_poses,image_paths,plot_imgs_paths
        else:
            return inputs, labels, selected_idx,future_speed_and_curvatur

class Edges2CloudMultiFrameDataset(Dataset):

    def __init__(self, sample_root,input_file, check_path,processor,client_num,dev=None,model_name=None):
        self.sample_root=sample_root
        self.client_num = client_num
        with open(input_file,'r') as f:
            self.data = json.load(f)
        self.teachers_data=[]
        for client_idx in range(self.client_num):
            client_input_file_path=os.path.join(check_path.replace('Cloud',f'Client{client_idx}'),'distil_cloud_train_data',os.path.basename(input_file).replace('json','pt'))
            self.teachers_data.append(torch.load(client_input_file_path,map_location='cpu', weights_only=False))
            # with open(client_input_file_path,'r') as f:
            #     self.teachers_data.append(json.load(f))
        self.processor=processor
        self.tokenizer = processor.tokenizer
        self.dev=dev

        # self.model_name=model_name


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'img_path':[os.path.join(self.sample_root, self.data[idx]['current_imgs'][i]).replace('\\','/') for i in range(len(self.data[idx]['current_imgs']))],
            'obs_speed_and_curvatur_str':self.data[idx]['obs_speed_and_curvatur']['str'],
            'obs_speed_and_curvatur': self.data[idx]['obs_speed_and_curvatur']['float'],
            'future_speed_and_curvatur_str':self.data[idx]['future_speed_and_curvatur']['str'],
            'future_speed_and_curvatur': self.data[idx]['future_speed_and_curvatur']['float'],

            'obs_trajs_str': self.data[idx]['obs_trajs']['str'],
            'obs_trajs': self.data[idx]['obs_trajs']['float'],
            'future_trajs_str': self.data[idx]['future_trajs']['str'],
            'future_trajs': self.data[idx]['future_trajs']['float'],

            'selected_idx':idx,
            'fut_ego_traj_world':self.data[idx]['eval_meta']['fut_ego_traj_world'],
            'fut_start_world': self.data[idx]['eval_meta']['fut_start_world'],
            'obs_ego_velocities': self.data[idx]['eval_meta']['obs_ego_velocities'],
            'teachers_personal_features': [self.teachers_data[client_idx][idx]['last_peft_output']['personal'] for client_idx in range(self.client_num)],
            'teachers_class': torch.tensor([int(client_idx + 1) for client_idx in range(self.client_num)]),

            'teachers_res_logit':[self.teachers_data[client_idx][idx]['local_output'] for client_idx in range(self.client_num)]
        }


    def custom_collate_fn(self,features):
        # image_paths = [f["img_path"] for f in features]
        # his_speed_and_curvatur_str = [f["his_speed_and_curvatur_str"] for f in features]
        # future_speed_and_curvatur_str = [f["future_speed_and_curvatur_str"] for f in features]
        selected_idx = [f['selected_idx'] for f in features]
        # fut_ego_traj_world=[f["fut_ego_traj_world"] for f in features]
        # fut_start_world = [f["fut_start_world"] for f in features]
        # obs_ego_velocities = [f["obs_ego_velocities"] for f in features]
        # teachers_logits=[f['teachers_logits'] for f in features]
        teachers_personal_features=[f['teachers_personal_features'] for f in features]
        teachers_class=[f['teachers_class'] for f in features]
        future_speed_and_curvatur = [f["future_speed_and_curvatur"] for f in features]
        future_trajs = [f["future_trajs"] for f in features]
        teachers_res_logit=[f['teachers_res_logit'] for f in features]

        prompts = []
        full_texts = []
        messages_list = []

        for idx in range(len(features)):
            prompt_text = (
                f"You are a autonomous driving assistant.You are provided with a sequence of 10 front-view camera images collected over the past 5 seconds. The corresponding speeds and curvatures of the ego vehicle at each time step are:{features[idx]['obs_speed_and_curvatur_str']}. Note: All speeds are normalized using min-max normalization and curvatures are normalized using Z-score normalization.\n "
                f"Your task is to analyze the driving history and the current frame (the last frame of the front-view camera images sequences) to predict future vehicle speeds and curvatures."
                f"Think aloud step by step:\n"
                f"(1) Scene understanding: Describe traffic lights, other vehicles, pedestrians, and lane markings.\n"
                f"(2) Risk awareness: Are there objects or agents the ego car should be cautious about?\n"
                f"(3) Driving intention: Based on the motion history and current view, is the car going straight, turning, slowing, or accelerating?\\n"

                f"Finally, predict the next 10 [speed, curvature] values of the ego car.\n"
                f"Format: [speed_1, curvature_1], [speed_2, curvature_2], ..., [speed_10, curvature_10]. Each value should be a float rounded to four decimal places."
            )
            label_text = features[idx]['future_speed_and_curvatur_str']
            # 构建消息体
            messages = [{
                "role": "user",
                "content": [
                    *[
                        {"type": "image", "image": f'file://{features[idx]["img_path"][i]}'} for i in
                        range(len(features[idx]["img_path"]))
                    ],
                    {"type": "text", "text": prompt_text}
                ]
            }]
            messages_list.append(messages)
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            full_texts.append(prompt + label_text)

        image_inputs, video_inputs = process_vision_info(messages_list)

        # 使用 prompt + label_text 作为模型的输入
        inputs = self.processor(
            text=full_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        # 使用 tokenizer 构造 label，但把 prompt 部分 mask 掉，只保留 target 部分参与训练
        labels = inputs["input_ids"].clone()
        for i, prompt in enumerate(prompts):
            prompt_len = len(self.tokenizer(prompt).input_ids)
            labels[i, :prompt_len] = -100  # mask prompt 部分


        return inputs,labels,teachers_res_logit,teachers_personal_features,torch.stack(teachers_class),future_speed_and_curvatur, future_trajs,selected_idx

class Cloud2EdgesMultiFrameDataset(Dataset):

    def __init__(self, sample_root,input_file, processor,client_idx,check_path,communication,dev=None,model_name=None):
        self.sample_root=sample_root
        self.input_file=input_file

        with open(input_file,'r') as f:
            self.data = json.load(f)

        self.teacher_path=os.path.join(check_path.replace(f'Client{client_idx}','Cloud'),'distil_cloud_train_data',f'{communication-1}_train.pt')
        self.teachers_data=torch.load(self.teacher_path,map_location='cpu', weights_only=False)
        self.check_effective_data()
        # with open(teacher_path,'r') as f:
        #     self.teachers_data=json.load(f)
        self.processor=processor
        self.tokenizer = processor.tokenizer
        self.dev=dev
        # self.model_name=model_name
    def check_effective_data(self):
        sample_num=self.__len__()
        effective_data=[]
        effective_teachers_data=[]
        for idx in range(sample_num):
            if self.teachers_data[idx]['local_output'] is not None:
                effective_data.append(self.data[idx])
                effective_teachers_data.append(self.teachers_data[idx])
        self.data=effective_data
        self.teachers_data=effective_teachers_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'img_path':[os.path.join(self.sample_root, self.data[idx]['current_imgs'][i]).replace('\\','/') for i in range(len(self.data[idx]['current_imgs']))],
            'obs_speed_and_curvatur_str':self.data[idx]['obs_speed_and_curvatur']['str'],
            'obs_speed_and_curvatur': self.data[idx]['obs_speed_and_curvatur']['float'],
            'future_speed_and_curvatur_str':self.data[idx]['future_speed_and_curvatur']['str'],
            'future_speed_and_curvatur': self.data[idx]['future_speed_and_curvatur']['float'],

            'obs_trajs_str': self.data[idx]['obs_trajs']['str'],
            'obs_trajs': self.data[idx]['obs_trajs']['float'],
            'future_trajs_str': self.data[idx]['future_trajs']['str'],
            'future_trajs': self.data[idx]['future_trajs']['float'],

            'selected_idx':idx,
            'fut_ego_traj_world':self.data[idx]['eval_meta']['fut_ego_traj_world'],
            'fut_start_world': self.data[idx]['eval_meta']['fut_start_world'],
            'obs_ego_velocities': self.data[idx]['eval_meta']['obs_ego_velocities'],
            'teachers_logits':self.teachers_data[idx]['local_output'],
            # 'teachers_gen_feat':self.teachers_data[idx]['last_peft_output']['global']+self.teachers_data[idx]['last_peft_output']['w_output'],
            # 'teachers_gen_feat': self.teachers_data[idx]['last_peft_output']['pooled_features']
        }




    def custom_collate_fn(self,features):
        # obs_imgs,obs_speed_and_curvatur,future_speed_and_curvatur = zip(*batch)
        # image_paths = [f["img_path"] for f in features]
        # his_speed_and_curvatur_str = [f["his_speed_and_curvatur_str"] for f in features]
        # future_speed_and_curvatur_str = [f["future_speed_and_curvatur_str"] for f in features]
        teachers_logits=[f['teachers_logits'] for f in features]
        # teachers_gen_features=[f['teachers_gen_feat'] for f in features]
        # fut_ego_traj_world=[f["fut_ego_traj_world"] for f in features]
        # fut_start_world = [f["fut_start_world"] for f in features]
        # obs_ego_velocities = [f["obs_ego_velocities"] for f in features]
        future_speed_and_curvatur = [f["future_speed_and_curvatur"] for f in features]
        future_trajs = [f["future_trajs"] for f in features]


        prompts = []
        full_texts = []
        messages_list = []
        labels_texts = []

        for idx in range(len(features)):
            prompt_text = (
                f"You are a autonomous driving assistant.You are provided with a sequence of 10 front-view camera images collected over the past 5 seconds. The corresponding speeds and curvatures of the ego vehicle at each time step are:{features[idx]['obs_speed_and_curvatur_str']}. Note: All speeds are normalized using min-max normalization and curvatures are normalized using Z-score normalization.\n "
                f"Your task is to analyze the driving history and the current frame (the last frame of the front-view camera images sequences) to predict future vehicle speeds and curvatures."
                f"Think aloud step by step:\n"
                f"(1) Scene understanding: Describe traffic lights, other vehicles, pedestrians, and lane markings.\n"
                f"(2) Risk awareness: Are there objects or agents the ego car should be cautious about?\n"
                f"(3) Driving intention: Based on the motion history and current view, is the car going straight, turning, slowing, or accelerating?\\n"

                f"Finally, predict the next 10 [speed, curvature] values of the ego car.\n"
                f"Format: [speed_1, curvature_1], [speed_2, curvature_2], ..., [speed_10, curvature_10]. Each value should be a float rounded to four decimal places."
            )
            label_text = features[idx]['future_speed_and_curvatur_str']
            # 构建消息体
            messages = [{
                "role": "user",
                "content": [
                    *[
                        {"type": "image", "image": f'file://{features[idx]["img_path"][i]}'} for i in
                        range(len(features[idx]["img_path"]))
                    ],
                    {"type": "text", "text": prompt_text}
                ]
            }]
            messages_list.append(messages)
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            full_texts.append(prompt + label_text)
            labels_texts.append(label_text)

        image_inputs, video_inputs = process_vision_info(messages_list)

        # 使用 prompt + label_text 作为模型的输入
        inputs = self.processor(
            text=full_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        labels = inputs["input_ids"].clone()
        for i, prompt in enumerate(prompts):
            prompt_len = len(self.tokenizer(prompt).input_ids)
            labels[i, :prompt_len] = -100  # mask prompt 部分
        # 使用 tokenizer 构造 label，但把 prompt 部分 mask 掉，只保留 target 部分参与训练
        # labels = inputs["input_ids"].clone()
        # for i, (prompt, label_text) in enumerate(zip(prompts, labels_texts)):
        #     prompt_ids = self.tokenizer(prompt).input_ids
        #     label_ids = self.tokenizer(label_text).input_ids
        #
        #     prompt_len = len(prompt_ids)
        #     total_len = len(inputs["input_ids"][i])
        #
        #     # mask 掉 prompt 之前的部分，仅保留 label_text 参与 loss
        #     labels[i, :prompt_len] = -100
        #
        #     # 如果 tokenizer 会插入 <eos>，可能总长度会略大于 prompt_len + len(label_ids)，建议加一层保险：
        #     labels[i, prompt_len + len(label_ids):] = -100

        return inputs,labels,torch.stack(teachers_logits),future_speed_and_curvatur,future_trajs

