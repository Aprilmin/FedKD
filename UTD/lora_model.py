from multimodal_module import FeatureFusionModel
from PIL import Image
import requests
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel,CLIPTokenizer
import torch
import os
import copy
import numpy as np

from torch.autograd import Function



def loadLLM(pretrained_model_path):
    model_name=os.path.basename(pretrained_model_path)
    processor = CLIPProcessor.from_pretrained(pretrained_model_path)
    tokenizer=CLIPTokenizer.from_pretrained(pretrained_model_path)
    model = CLIPModel.from_pretrained(pretrained_model_path)
    return model,processor,tokenizer


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


class MultimodalCLIP(nn.Module):
    def __init__(self,pretrained_model,pretrained_model_path=r'D:\PycharmProjects\mycode\2025-DoubleCap\UTD\models\clip-vit-large-patch14'):
        super(MultimodalCLIP, self).__init__()
        self.clip_model=pretrained_model

        feature_out_num=768 if 'clip-vit-large-patch14' in pretrained_model_path else 512
        self.multimodal_fusion_model=FeatureFusionModel(hiddenNum=512,fusion_rank=5,feature_out_num=feature_out_num)


class DriveMLLM(nn.Module):
    def __init__(self,rank=8,lora_alpha=32,peft='pertucker',pretrained_model_path=r'D:\PycharmProjects\mycode\2025-DoubleCap\UTD\models\clip-vit-large-patch14'):
        super(DriveMLLM, self).__init__()
        pretrained_model, self.processor, self.tokenizer = loadLLM(pretrained_model_path)
        for param in pretrained_model.parameters():
            param.requires_grad=False
        self.peft=peft
        self.model_name = os.path.basename(pretrained_model_path)
        self.model=MultimodalCLIP(pretrained_model,pretrained_model_path)
        # print_trainable_parameters(self.model)

        if peft.lower() in ['pertucker','tlora']:
            self.shared_mgr = self.SharedParamsManager(self.model_name, peft, rank, pretrained_model.device)
            module_num, self.model.clip_model.text_model = self.inject_lora(peft_name=peft, model=self.model.clip_model.text_model, shared_mgr=self.shared_mgr,target_modules=['q_proj', 'v_proj'],lora_alpha=lora_alpha, rank=rank, get_shared_module_method=self._get_shared_module,idx_counter=[0])
        elif 'lora' in peft:
            module_num, self.model.clip_model.text_model = self.inject_lora(peft, self.model.clip_model.text_model, shared_mgr=None,target_modules=['q_proj', 'v_proj'],lora_alpha=lora_alpha, rank=rank, idx_counter=[0])



        self.last_peft_layer,self.last_peft_layer_name = self.find_last_peft_layer()

    def print_trainable_peft_param(self):
        trainable_param =0
        all_param=0
        for name,param in self.model.clip_model.text_model.named_parameters():
            all_param+=param.numel()
            if param.requires_grad:
                trainable_param+=param.numel()
        if self.peft.lower() in ['pertucker','tlora']:
            for name,param in self.shared_mgr.named_parameters():
                trainable_param+=param.numel()
        print(
            f"Trainable params: {trainable_param} || all params: {all_param} || trainable%: {100 * trainable_param / all_param}"
        )

    def forward(self,inputData,subnetFlag,texts,device,return_components=False):
        fused_feat,_,_=self.model.multimodal_fusion_model(inputData,subnetFlag)
        text_inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(device)
        text_feat = self.model.clip_model.get_text_features(**text_inputs)  # 不加 no_grad
        fused_feat = torch.nn.functional.normalize(fused_feat, dim=-1)
        text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
        logits = fused_feat @ text_feat.T

        if return_components:
            return fused_feat,text_feat,logits
        else:
            return logits


    def set_personal_disable(self,model):
        personal_params_for_recover={}
        for name,param in model.named_parameters():
            if 'personal' in name and 'global' not in name:
                personal_params_for_recover[name]=copy.deepcopy(param.data.detach().cpu())
                param.data=torch.zeros_like(param.data)
                param.requires_grad=False
        return personal_params_for_recover
    def set_personal_able(self,model,personal_params_for_recover):
        for name,param in model.named_parameters():
            if name in personal_params_for_recover.keys():
                param.data=copy.deepcopy(personal_params_for_recover[name])
                param.requires_grad=True

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
        """统计模型中所有需要注入的模块数量"""
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
        """遍历模型，替换所有线性层为perTuckerLayer"""
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

            self.linear = linear_layer #linear_layer  # 原始线性层（冻结）
            for param in self.linear.parameters():
                param.requires_grad = False

            self.global_A_matric = nn.Parameter(torch.zeros(rank, self.linear.in_features))  # Ag ∈ ℝ^{r×k}
            self.global_B_matric = nn.Parameter(torch.zeros(self.linear.out_features, rank))

            # if random_init_A:
            nn.init.constant_(self.global_A_matric, 0.0)
            # if random_init_B:
            nn.init.normal_(self.global_B_matric, mean=0, std=0.02)  # Ag随机初始化

            self.scaling = lora_alpha / rank

            self.global_A_matric.requires_grad_(True)
            self.global_B_matric.requires_grad_(True)


        def forward(self, x):

            _g_x=x @ self.global_A_matric.to(dtype=x.dtype,device=x.device).T @ self.global_B_matric.to(dtype=x.dtype,device=x.device).T
            _w_x=self.linear(x)
            self._g_x=_g_x
            self._w_x=_w_x
            # print(f'###########{self.linear.__class__}#####{self.linear.forward}#####{self.linear}, {_w_x.shape}, {x.shape}###########{traceback.print_stack(limit=5)}#########')
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
            self._g_x = x @ self.get_shared_module_method(self.target_name,'shared_global').A.to(dtype=x.dtype,device=x.device).T @ self.global_U.to(dtype=x.dtype,device=x.device).T @ self.get_shared_module_method(self.target_name,'shared_global').V.to(dtype=x.dtype, device=x.device).T
            self._p_x = x @ self.get_shared_module_method(self.target_name,'shared_personal').A.to(dtype=x.dtype,device=x.device).T @ self.personal_U.to(dtype=x.dtype,device=x.device).T @ self.get_shared_module_method(self.target_name,'shared_personal').V.to(dtype=x.dtype, device=x.device).T
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
            self._g_x = x @ self.get_shared_module_method(self.target_name,'shared_global').A.to(dtype=x.dtype,device=x.device).T @ self.global_U.to(dtype=x.dtype,device=x.device).T @ self.get_shared_module_method(self.target_name,'shared_global').V.to(dtype=x.dtype, device=x.device).T
            self._w_x = self.linear(x)
            return self._w_x + (self._g_x) * self.scaling
    class SharedParamsManager(nn.Module):
        """集中管理所有PEFT变体的共享参数"""

        def __init__(self, model_name, peft_type, rank, device):
            super().__init__()
            if model_name in ['clip-vit-base-patch32']:
                q_in, q_out, v_in, v_out = 512, 512, 512, 512
            elif model_name in ['clip-vit-large-patch14']:
                q_in, q_out, v_in, v_out = 768, 768, 768, 768
            self.peft_type = peft_type
            self.rank = rank

            # 管理引用关系（不是注册，只是索引）




            self.sharedMatrics = nn.ModuleDict({
                'q_proj': nn.ModuleDict({
                    'shared_global': self.SharedMatircsParams(q_in, q_out, rank, device),
                    'shared_personal': self.SharedMatircsParams(q_in, q_out, rank, device) if self.peft_type == 'pertucker' else None
                }),
                'v_proj': nn.ModuleDict({
                    'shared_global': self.SharedMatircsParams(v_in, v_out, rank, device),
                    'shared_personal': self.SharedMatircsParams(v_in, v_out, rank, device) if self.peft_type == 'pertucker' else None
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

    def freeze_block(self,model,freeze_personal_block=False,freeze_global_block=False,freeze_shared_block=False,freeze_multimodal_fusion=False):
        for name,param in model.named_parameters():
            if 'personal' in name and 'global' not in name:
                param.requires_grad=False if freeze_personal_block else True
            elif 'global' in name and 'personal' not in name:
                param.requires_grad=False if freeze_global_block else True
            elif 'global' in name and 'personal' in name:
                param.requires_grad = False if freeze_shared_block else True
            elif 'multimodal_fusion_model' in name:
                param.requires_grad = False if freeze_multimodal_fusion else True
            else:
                param.requires_grad = False
class Edge2CloudDistil(DriveMLLM):
    def __init__(self,rank=8,lora_alpha=32,peft='pertucker',pretrained_model_path=r'D:\\PycharmProjects\\mycode\\2025-DoubleCap\\nuscenes\\models\\Qwen2-VL-2B-Instruct',
                 client_num=5,mapping_dim1=1,mapping_dim2=512,lambda_grl=1.0,temperature=1,is_test=False):
        super(Edge2CloudDistil, self).__init__(rank,lora_alpha,peft,pretrained_model_path)
        self.loss_fn = UncertaintyLoss()
        self.mapping_dim1=mapping_dim1
        self.mapping_dim2=mapping_dim2
        self.discriminator=DomainDiscriminator(mapping_dim1,mapping_dim2,num_domains=client_num)
        self.client_num=client_num

        self.temperature=temperature

    def forward(self,inputData,subnetFlag,texts,device,return_components=False,teachers_personal_feat=None,teacher_fusion_feat=None,teacher_text_feat=None,teachers_class=None,step_ratio=0,kl_gate=False,teacher_num=0):
        fused_feat,_,_=self.model.multimodal_fusion_model(inputData,subnetFlag)
        text_inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(device)
        text_feat = self.model.clip_model.get_text_features(**text_inputs)
        fused_feat = torch.nn.functional.normalize(fused_feat, dim=-1)
        text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
        logits = fused_feat @ text_feat.T
        if teachers_personal_feat is not None and teachers_class is not None:
            bz = len(teachers_personal_feat)
            lambda_grl = 2. / (1. + np.exp(-10 * step_ratio)) - 1 if kl_gate else 0.0
            pooled_personal_feat=torch.nn.functional.adaptive_avg_pool2d(self.last_peft_layer._g_x, output_size=(self.mapping_dim1, self.mapping_dim2)) # 27,1,512
            pooled_text_feat=torch.nn.functional.adaptive_avg_pool2d(text_feat.unsqueeze(1),output_size=(self.mapping_dim1, self.mapping_dim2))
            pooled_fused_feat = torch.nn.functional.adaptive_avg_pool2d(fused_feat.unsqueeze(1), output_size=(self.mapping_dim1, self.mapping_dim2))
            student_discirminator_input=torch.vstack([pooled_personal_feat,pooled_text_feat,pooled_fused_feat])

            teachers_discriminator_input=[]
            teachers_discriminator_target=[]
            for teacher_idx in range(teacher_num):
                teacher_pooled_personal_feat=torch.nn.functional.adaptive_avg_pool2d(teachers_personal_feat[0][teacher_idx], output_size=(self.mapping_dim1, self.mapping_dim2))
                teacher_pooled_text_feat=torch.nn.functional.adaptive_avg_pool2d(teacher_text_feat[0][teacher_idx].unsqueeze(1),output_size=(self.mapping_dim1, self.mapping_dim2))
                teacher_pooled_fused_feat=torch.nn.functional.adaptive_avg_pool2d(torch.stack([teacher_fusion_feat[sampleIdx][teacher_idx] for sampleIdx in range(bz)],dim=0).unsqueeze(1), output_size=(self.mapping_dim1, self.mapping_dim2))

                res_personal_feat=teacher_pooled_personal_feat-pooled_personal_feat
                res_text_feat=teacher_pooled_text_feat-pooled_text_feat
                res_fused_feat=teacher_pooled_fused_feat-pooled_fused_feat

                teacher_discirminator_input = torch.vstack([teacher_pooled_personal_feat, teacher_pooled_text_feat, teacher_pooled_fused_feat, res_personal_feat,res_text_feat, res_fused_feat])
                teacher_discirminator_target= torch.full((teacher_discirminator_input.shape[0],), teachers_class[teacher_idx],device=device)



                teachers_discriminator_input.append(teacher_discirminator_input)
                teachers_discriminator_target.append(teacher_discirminator_target)

            teachers_discriminator_input=torch.cat(teachers_discriminator_input,dim=0)
            teachers_discriminator_target=torch.cat(teachers_discriminator_target,dim=0).long()

            teacher_logits=self.discriminator(self.discriminator.attention_pool(teachers_discriminator_input))
            teacher_loss = torch.nn.functional.cross_entropy(teacher_logits, teachers_discriminator_target)

            student_logits=self.discriminator(self.discriminator.attention_pool(grad_reverse(student_discirminator_input, lambda_grl)))
            student_prob = torch.nn.functional.softmax(student_logits, dim=1)
            student_entropy = -torch.sum(student_prob * torch.log(student_prob + 1e-6), dim=1).mean()




            return teacher_loss, student_entropy,fused_feat,text_feat,logits


        else:
            if return_components:
                return fused_feat,text_feat,logits
            else:
                return logits









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
class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        # x: [B, seq_len, dim]
        # if bfloat:
        #     x = x.to(dtype=torch.bfloat16)
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

    def forward(self, x):
        # if bfloat:
        #     x = x.to(dtype=torch.bfloat16)
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



