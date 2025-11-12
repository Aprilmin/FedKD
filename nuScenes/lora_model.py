#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import transformers
import os
from torch.autograd import Function
import numpy as np
import re
import copy
class UncertaintyLoss(nn.Module):
    def __init__(self):
        super(UncertaintyLoss, self).__init__()
        # 初始化 log(sigma^2)，即 log-variance，初值为 0（相当于 sigma=1）
        self.log_sigma_speed = torch.nn.Parameter(torch.tensor(0.0))
        self.log_sigma_curv = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, speed_mse_loss, curv_mse_loss):
        loss = (
            torch.exp(-self.log_sigma_speed) * speed_mse_loss + self.log_sigma_speed +
            torch.exp(-self.log_sigma_curv) * curv_mse_loss + self.log_sigma_curv
        )
        return loss
class SpeedCurvatureHead(nn.Module):
    def __init__(self, hidden_size, horizon=10):
        super().__init__()
        if hidden_size==1536:

            self.shared = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )


            self.speed_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Linear(hidden_size//2, horizon),
                nn.Sigmoid()
            )



            self.curvature_head = nn.Sequential(
                nn.Linear(hidden_size, horizon),
                nn.Tanh()
            )
        elif hidden_size==2048:
            self.shared = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size)
            )
            self.speed_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, horizon),
                nn.Sigmoid()
            )
            self.curvature_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, horizon),
                nn.Tanh()
            )
        elif hidden_size==3584:
            self.shared = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(0.1)
            )
            self.speed_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, horizon),
                nn.Sigmoid()
            )
            self.curvature_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, horizon),
                nn.Tanh()
            )

    def forward(self, x):
        feat = self.shared.to(x.dtype)(x)
        speed = self.speed_head.to(x.dtype)(feat)
        curvature = self.curvature_head.to(x.dtype)(feat)
        out = torch.stack([speed, curvature], dim=-1)  # (B, T, 2)
        return out

class DriveMLLM(nn.Module):
    def __init__(self,rank=8,lora_alpha=32,peft='pertucker',pretrained_model_path=r'D:\\PycharmProjects\\mycode\\2025-DoubleCap\\nuscenes\\models\\Qwen2-VL-2B-Instruct',is_test=False):
        super(DriveMLLM, self).__init__()
        self.loss_fn = UncertaintyLoss()
        self.peft=peft
        self.model_name = os.path.basename(pretrained_model_path)
        self.model, self.processor = loadLLM(pretrained_model_path,is_test=is_test)
        for param in self.model.parameters():
            param.requires_grad=False



        print_trainable_parameters(self.model)

        if peft.lower() in ['pertucker','tlora']:
            self.shared_mgr = self.SharedParamsManager(self.model_name, peft, rank, self.model.device)
            module_num, self.model = self.inject_lora(peft_name=peft, model=self.model, shared_mgr=self.shared_mgr,target_modules=['q_proj', 'v_proj'],lora_alpha=lora_alpha, rank=rank, get_shared_module_method=self._get_shared_module,idx_counter=[0])
        elif 'lora' in peft:
            module_num, self.model = self.inject_lora(peft, self.model, shared_mgr=None,target_modules=['q_proj', 'v_proj'],lora_alpha=lora_alpha, rank=rank, idx_counter=[0])

        print('Trainable Parameters for {} model:'.format(peft))
        hidden_size=1536 if self.model_name in ['Qwen2-VL-2B-Instruct'] else 2048 if self.model_name in ['Qwen2.5-VL-3B-Instruct'] else 3584 if self.model_name in ['Qwen2.5-VL-7B-Instruct', 'Qwen2-VL-7B-Instruct'] else None
        self.global_head=SpeedCurvatureHead(hidden_size)
        print_trainable_parameters(self.model)

        self.last_peft_layer,self.last_peft_layer_name = self.find_last_peft_layer()

    def forward(self,inputs,labels=None):
        if labels is not None:
            inputs['labels'] = labels
        backbone_outputs=self.model(**inputs,output_hidden_states=True,return_dict=True )
        last_hidden=backbone_outputs.hidden_states[-1]
        pooled=last_hidden.mean(dim=1)
        head_outputs=self.global_head(pooled)
        return backbone_outputs,head_outputs,pooled

    def generate(self,inputs):
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_texts_trimmed = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_texts_trimmed
    def set_personal_disable(self,model):
        personal_params_for_recover={}
        for name,param in model.named_parameters():
            if 'personal' in name and 'global' not in name:
                personal_params_for_recover[name]=copy.deepcopy(param.data.detach().cpu())
                param.data=torch.zeros_like(param.data)
                param.requires_grad=False
        return personal_params_for_recover

    def extract_trajectory(self,text):
        try:
            # 只保留前10个预测对
            nums = re.findall(r'\[\s*[\d\.\-eE]+\s*,\s*[\d\.\-eE]+\s*\]', text)
            return [eval(n) for n in nums[:10]]
        except:
            return None


    def find_last_peft_layer(self):
        last_peft_layer = None
        last_peft_layer_name=None
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, self.TuckerLayer) or isinstance(module, self.LoraLayer) or isinstance(module, self.perTuckerLayer):
                last_peft_layer = module
                last_peft_layer_name=name
                break
        return last_peft_layer,last_peft_layer_name

    def _get_all_peft_params(self,model):
        saved_dict = {}
        for name, param in model.named_parameters():
            if (param.requires_grad) or 'personal' in name or 'global' in name or 'share' in name:
                saved_dict[name] = param.clone()
        return saved_dict



    def _get_shared_module(self,target_name,shared_type):
        return self.shared_mgr.sharedMatrics[target_name][shared_type]

    def count_target_modules(self,model, target_modules=['q_proj', 'v_proj']):
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                for target_name in target_modules:
                    if target_name in name:
                        count += 1
                        break
        return count

    def inject_lora(self, peft_name, model, shared_mgr=None, target_modules=['q_proj', 'v_proj'], lora_alpha=32, rank=64,get_shared_module_method=None,idx_counter=[0],total_count=None):
        if total_count is None:
            total_count = self.count_target_modules(model, target_modules)
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                for target_name in target_modules:
                    if target_name in name:
                        # if name in target_modules:
                        idx_counter[0] += 1
                        is_last = idx_counter[0] == total_count
                        idx_counter.append(name)
                        if peft_name.lower()=='pertucker':
                            adapter = self.perTuckerLayer(module,lora_alpha,get_shared_module_method,target_name)
                        elif peft_name.lower()=='tlora':
                            adapter = self.TuckerLayer(module,lora_alpha,get_shared_module_method,target_name)
                        elif 'lora' in peft_name:
                            adapter = self.LoraLayer(module, module.in_features, module.out_features, rank, lora_alpha)
                        else:
                            raise ValueError(f"Unsupported PEFT type: {peft_name}")
                        if is_last:
                            self.last_peft_layer = adapter
                        setattr(model, name, adapter)
            else:
                # 递归处理子模块
                self.inject_lora(peft_name, module, shared_mgr, target_modules, lora_alpha, rank, get_shared_module_method,idx_counter,total_count)

        # return idx_counter[0], model
        return idx_counter,model

    class LoraLayer(nn.Module):
        """替换原始线性层：Wx + Gx + Px"""

        def __init__(self, linear_layer, in_dim, out_dim, rank, lora_alpha,random_init_A=False, random_init_B=False,):
            super().__init__()

            self.linear = linear_layer
            for param in self.linear.parameters():
                param.requires_grad = False

            self.global_A_matric = nn.Parameter(torch.zeros(rank, self.linear.in_features))  # Ag ∈ ℝ^{r×k}
            self.global_B_matric = nn.Parameter(torch.zeros(self.linear.out_features, rank))

            # if random_init_A:

            nn.init.constant_(self.global_A_matric, 0.0)
            nn.init.normal_(self.global_B_matric, mean=0, std=0.02)  # Ag随机初始化

            self.scaling = lora_alpha / rank

            self.global_A_matric.requires_grad_(True)
            self.global_B_matric.requires_grad_(True)


        def forward(self, x):
            _g_x=x @ self.global_A_matric.to(dtype=x.dtype,device=x.device).T @ self.global_B_matric.to(dtype=x.dtype,device=x.device).T
            _w_x=self.linear(x)
            self._g_x=_g_x
            self._w_x=_w_x

            return  _w_x + _g_x * self.scaling

    class perTuckerLayer(nn.Module):
        def __init__(self, linear_layer, lora_alpha,get_shared_module_method,target_name):
            super().__init__()
            self.linear = linear_layer  # 原始线性层（冻结）
            for param in self.linear.parameters():
                param.requires_grad = False
            self.get_shared_module_method=get_shared_module_method
            self.target_name=target_name
            rank=self.get_shared_module_method(target_name,'shared_global').V.shape[1]


            self.global_U = nn.Parameter(torch.zeros(self.get_shared_module_method(target_name,'shared_global').V.shape[1], self.get_shared_module_method(target_name,'shared_global').A.shape[0]))
            self.global_U.requires_grad_(True)
            nn.init.normal_(self.global_U, mean=0, std=0.02)

            self.personal_U = nn.Parameter(torch.zeros(self.get_shared_module_method(target_name,'shared_personal').V.shape[1], self.get_shared_module_method(target_name,'shared_personal').A.shape[0]))
            self.personal_U.requires_grad_(True)
            nn.init.normal_(self.personal_U, mean=0, std=0.02)

            self.scaling = lora_alpha / rank


        def forward(self, x):
            self._g_x = x @ self.get_shared_module_method(self.target_name,'shared_global').A.to(dtype=x.dtype,device=x.device).T @ self.global_U.T @ self.get_shared_module_method(self.target_name,'shared_global').V.to(dtype=x.dtype, device=x.device).T
            self._p_x = x @ self.get_shared_module_method(self.target_name,'shared_personal').A.to(dtype=x.dtype,device=x.device).T @ self.personal_U.T @ self.get_shared_module_method(self.target_name,'shared_personal').V.to(dtype=x.dtype, device=x.device).T
            self._w_x = self.linear(x)

            # 实际输出
            return self._w_x + (self._g_x + self._p_x) * self.scaling


    class TuckerLayer(nn.Module):
        """替换原始线性层：Wx + Gx + Px"""

        def __init__(self, linear_layer, lora_alpha,get_shared_module_method,target_name):
            super().__init__()
            self.linear = linear_layer
            for param in self.linear.parameters():
                param.requires_grad = False
            self.get_shared_module_method=get_shared_module_method
            self.target_name=target_name
            rank=self.get_shared_module_method(target_name,'shared_global').V.shape[1]

            self.global_U = nn.Parameter(torch.zeros(self.get_shared_module_method(target_name,'shared_global').V.shape[1], self.get_shared_module_method(target_name,'shared_global').A.shape[0]))
            self.global_U.requires_grad_(True)
            nn.init.normal_(self.global_U, mean=0, std=0.02)
            self.scaling = lora_alpha / rank

        def forward(self, x):
            self._g_x = x @ self.get_shared_module_method(self.target_name,'shared_global').A.to(dtype=x.dtype,device=x.device).T @ self.global_U.T @ self.get_shared_module_method(self.target_name,'shared_global').V.to(dtype=x.dtype, device=x.device).T
            self._w_x = self.linear(x)
            return self._w_x + (self._g_x) * self.scaling
    class SharedParamsManager(nn.Module):
        """集中管理所有PEFT变体的共享参数"""

        def __init__(self, model_name, peft_type, rank, device):
            super().__init__()
            if model_name in ['Qwen2.5-VL-7B-Instruct', 'Qwen2-VL-7B-Instruct']:
                q_in, q_out, v_in, v_out = 3584, 3584, 3584, 512
            elif model_name in ['Qwen2.5-VL-3B-Instruct']:
                q_in, q_out, v_in, v_out = 2048, 2048, 2048, 256
            elif model_name in ['Qwen2-VL-2B-Instruct']:
                q_in, q_out, v_in, v_out = 1536, 1536, 1536, 256
            elif 't5' in model_name.lower():
                q_in, q_out, v_in, v_out = 768, 768, 768, 768
            self.peft_type = peft_type
            self.rank = rank

            # 管理引用关系（不是注册，只是索引）




            self.sharedMatrics = nn.ModuleDict({
                'q_proj': nn.ModuleDict({
                    'shared_global': self.SharedMatircsParams(q_in, q_out, rank, device),
                    'shared_personal': self.SharedMatircsParams(q_in, q_out, rank, device) if self.peft_type =='pertucker' else None
                }),
                'v_proj': nn.ModuleDict({
                    'shared_global': self.SharedMatircsParams(v_in, v_out, rank, device),
                    'shared_personal': self.SharedMatircsParams(v_in, v_out, rank, device) if self.peft_type =='pertucker' else None
                })
            })




        class SharedMatircsParams(nn.Module):
            def __init__(self, in_dim, out_dim, rank, device):
                super().__init__()
                self.A = nn.Parameter(torch.zeros(rank, in_dim))  # Ag ∈ ℝ^{r×k}
                self.V = nn.Parameter(torch.zeros(out_dim, rank))  # Vg ∈ ℝ^{d×r}


                # nn.init.constant_(self.A, 0.0)
                nn.init.normal_(self.V, mean=0, std=0.02)


                nn.init.normal_(self.A, mean=0, std=0.02)
                # nn.init.kaiming_normal_(self.V, mode='fan_out', nonlinearity='linear')
                self.A.requires_grad_(True)
                self.V.requires_grad_(True)
                self.A.to(device)
                self.V.to(device)

    def freeze_block(self,model,freeze_personal_block=False,freeze_global_block=False,freeze_shared_block=False):
        for name,param in model.named_parameters():
            if 'personal' in name and 'global' not in name:
                param.requires_grad=False if freeze_personal_block else True
            elif 'global' in name and 'personal' not in name:
                param.requires_grad=False if freeze_global_block else True
            elif 'global' in name and 'personal' in name:
                param.requires_grad = False if freeze_shared_block else True
            else:
                param.requires_grad = False



class Edge2CloudDistil(DriveMLLM):
    def __init__(self,rank=8,lora_alpha=32,peft='pertucker',pretrained_model_path=r'D:\\PycharmProjects\\mycode\\2025-DoubleCap\\nuscenes\\models\\Qwen2-VL-2B-Instruct',
                 client_num=2,mapping_dim1=2048,mapping_dim2=256,lambda_grl=1.0,temperature=1,is_test=False):
        super(Edge2CloudDistil, self).__init__(rank,lora_alpha,peft,pretrained_model_path,is_test)
        self.loss_fn = UncertaintyLoss()
        self.mapping_dim1=mapping_dim1
        self.mapping_dim2=mapping_dim2
        self.discriminator=DomainDiscriminator(mapping_dim1,mapping_dim2,num_domains=client_num)

        self.temperature=temperature


    def forward(self,inputs,labels,teachers_personal_features=None,teachers_class=None,kl_gate=False,step_ratio=0,bfloat=False):
        inputs['labels'] = labels
        cloud_logit=self.model(**inputs,output_hidden_states=True,return_dict=True)
        last_hidden=cloud_logit.hidden_states[-1]
        pooled = last_hidden.mean(dim=1)
        head_outputs=self.global_head(pooled)
        kl_loss=0

        if teachers_personal_features is not None and teachers_class is not None:
            q_a_c=torch.nn.functional.adaptive_avg_pool2d(self.last_peft_layer._g_x,output_size=(self.mapping_dim1, self.mapping_dim2))  # [bz,mapping_dim1,mapping_dim2]

            client_num=len(teachers_personal_features[0])
            bz=len(teachers_personal_features)

            if kl_gate:
                lambda_grl = 2. / (1. + np.exp(-10 * step_ratio)) - 1
            else:
                lambda_grl = 0.0

            # z=torch.tanh(z)
            discriminator_input_t=[]
            discriminator_label_t=[]
            pooled_teachers_personal_feature=[]
            q_a_c = torch.nn.functional.normalize(q_a_c, dim=1)

            for client_idx in range(client_num):
                pooled_teacher_feature = torch.stack([torch.nn.functional.normalize(torch.nn.functional.adaptive_avg_pool2d(teachers_personal_features[sample_idx][client_idx].unsqueeze(0),output_size=(self.mapping_dim1, self.mapping_dim2)).squeeze(),dim=1) for sample_idx in range(bz)],dim=0)
                pooled_teachers_personal_feature.append(pooled_teacher_feature)

                discriminator_input_t.append(torch.nn.functional.normalize(pooled_teacher_feature - q_a_c, dim=1))
                discriminator_label_t.append(torch.full((bz,), client_idx, device=q_a_c.device))

                discriminator_input_t.append(torch.nn.functional.normalize(pooled_teacher_feature, dim=1))
                discriminator_label_t.append(torch.full((bz,), client_idx, device=q_a_c.device))
            discriminator_input_t = torch.cat(discriminator_input_t, dim=0)
            teacher_label = torch.cat(discriminator_label_t, dim=0).long()
            teacher_attention_pooled=self.discriminator.attention_pool(discriminator_input_t,bfloat)
            teacher_logits=self.discriminator(teacher_attention_pooled,bfloat)

            q_a_c_adv = grad_reverse(q_a_c, lambda_grl)
            student_logits=self.discriminator(self.discriminator.attention_pool(q_a_c_adv,bfloat),bfloat)


            teacher_loss = torch.nn.functional.cross_entropy(teacher_logits, teacher_label)
            student_prob = torch.nn.functional.softmax(student_logits, dim=1)
            student_entropy = -torch.sum(student_prob * torch.log(student_prob + 1e-6), dim=1).mean()




            ce_loss=cloud_logit.loss




            return head_outputs,ce_loss,kl_loss,teacher_loss,student_entropy,pooled
        else:
            return cloud_logit,head_outputs,pooled



class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x,bfloat):
        # x: [B, seq_len, dim]
        if bfloat:
            x = x.to(dtype=torch.bfloat16)
        attn_weights = self.attn(x)   # [B, seq_len, 1]
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=1)  # along seq_len
        weighted_sum = (x * attn_weights).sum(dim=1)  # [B, dim]
        return weighted_sum

class DomainDiscriminator(nn.Module):
    def __init__(self, mapping_dim1,mapping_dim2, hidden_dim=512, num_domains=3):
        super().__init__()
        input_dim=mapping_dim1*mapping_dim2
        self.net = nn.Sequential(
            nn.Linear(mapping_dim2, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_domains)  # e.g., 3 teachers + 1 student = 4 domains
        )
        self.attention_pool=AttentionPooling(input_dim=mapping_dim2)

    def forward(self, x,bfloat):
        if bfloat:
            x = x.to(dtype=torch.bfloat16)
        return self.net(x)

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None
def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)



def print_trainable_parameters(model,returnValue=False):
    """
    Prints the number of trainable parameters in the model
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"Trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    if returnValue:
        return trainable_params,all_param,100 * trainable_params / all_param





def loadLLM(pretrained_model_path, is_test=False):
    model,processor=None,None
    model_name=os.path.basename(pretrained_model_path)
    processor = transformers.AutoProcessor.from_pretrained(pretrained_model_path)
    if model_name in ['Qwen2.5-VL-3B-Instruct','Qwen2.5-VL-7B-Instruct']:
        model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained_model_path,use_cache=False) if not is_test \
            else transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained_model_path,device_map='auto',torch_dtype=torch.bfloat16)
    elif model_name in ['Qwen2-VL-2B-Instruct','Qwen2-VL-7B-Instruct']:
        model = transformers.Qwen2VLForConditionalGeneration.from_pretrained(pretrained_model_path,use_cache=False) if not is_test \
            else transformers.Qwen2VLForConditionalGeneration.from_pretrained(pretrained_model_path,device_map='auto',torch_dtype=torch.bfloat16)
    elif 't5' in model_name.lower():
        architectures='google-t5/t5-base' if 't5-base' in model_name.lower() else 'google-t5/t5-large' if 't5-large' in model_name.lower() else None
        # model = transformers.T5ForConditionalGeneration.from_pretrained(architectures,trust_remote_code=True,device_map='auto',torch_dtype=torch.bfloat16)
        model = transformers.T5ForConditionalGeneration.from_pretrained(architectures, trust_remote_code=True)
        processor = transformers.T5Tokenizer.from_pretrained(architectures)
    else:
        print('please add {}'.format(model_name))
    model.gradient_checkpointing_enable()
    model._set_gradient_checkpointing(enable=True,gradient_checkpointing_func=transformers.modeling_utils.checkpoint)
    return model,processor




