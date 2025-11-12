

python main.py --cloud_model_path '/data/gm/DoubleCap/models/Qwen2-VL-2B-Instruct' --edge_model_path '/data/gm/DoubleCap/models/Qwen2-VL-2B-Instruct' '/data/gm/DoubleCap/models/Qwen2-VL-2B-Instruct' --aggregation doublecap --peft perTucker --cloud_peft Tlora --gcs 2 --edge_bs 2 2 2 --rank 32 --gpus 7 --iid 0.1 --epochs 2 --cloud_epochs 4 --participate_data_ratio 0.5 --communications 10 --edge_alpha 0.3 --lambda_lm 0.01 --edge_lr 0.05 0.05 --d2cloud_lr 0.05 --d2edge_lr 0.05
python main.py --cloud_model_path '/data/gm/DoubleCap/models/Qwen2.5-VL-7B-Instruct' --edge_model_path '/data/gm/DoubleCap/models/Qwen2-VL-2B-Instruct' '/data/gm/DoubleCap/models/Qwen2-VL-2B-Instruct' --aggregation doublecap --peft perTucker --cloud_peft Tlora --gcs 2 --edge_bs 2 2 1 --rank 32 --gpus 7 --iid 0.1 --epochs 2 --cloud_epochs 4 --participate_data_ratio 0.5 --communications 10 --edge_alpha 0.3 --lambda_lm 0.01 --edge_lr 0.05 0.05 --d2cloud_lr  0.05 --d2edge_lr 0.05
python main.py --cloud_model_path '/data/gm/DoubleCap/models/Qwen2.5-VL-7B-Instruct' --edge_model_path '/data/gm/DoubleCap/models/Qwen2-VL-2B-Instruct' '/data/gm/DoubleCap/models/Qwen2.5-VL-3B-Instruct' --aggregation doublecap --peft perTucker --cloud_peft Tlora --gcs 2 --edge_bs 2 1 1 --rank 32 --gpus 7 --iid 0.1 --epochs 2 --cloud_epochs 4 --participate_data_ratio 0.5 --communications 10 --edge_alpha 0.3 --lambda_lm 0.01 --edge_lr 0.05 0.05 --d2cloud_lr  0.05 --d2edge_lr 0.05



