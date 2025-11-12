python main.py --aggregation doublecap --ModalityMixFrac 0 0 1 --iid 0.1 --rank 32 --client_num 10 --cloud_model_type large --edge_model_large_frac 0.4 --peft perTucker --cloud_peft Tlora --edge_alpha 0.3 --d2cloud_lr 0.005 --d2edge_lr 0.005 --startC 0 --endC 50
python main.py --aggregation doublecap --ModalityMixFrac 0 0 1 --iid 0.1 --rank 32 --client_num 10 --cloud_model_type large --edge_model_large_frac 0 --peft perTucker --cloud_peft Tlora --edge_alpha 0.3 --d2cloud_lr 0.005 --d2edge_lr 0.005 --startC 0 --endC 50
python main.py --aggregation doublecap --ModalityMixFrac 0 0 1 --iid 0.1 --rank 32 --client_num 10 --cloud_model_type large --edge_model_large_frac 1 --peft perTucker --cloud_peft Tlora --edge_alpha 0.3 --d2cloud_lr 0.005 --d2edge_lr 0.005 --startC 0 --endC 50


python main.py --aggregation doublecap --ModalityMixFrac 0 0.6 0.4 --iid 0.1 --rank 32 --client_num 10 --cloud_model_type large --edge_model_large_frac 0.4 --peft perTucker --cloud_peft Tlora --edge_alpha 0.3 --d2cloud_lr 0.005 --d2edge_lr 0.005 --startC 0 --endC 50
python main.py --aggregation doublecap --ModalityMixFrac 0 0.6 0.4 --iid 0.1 --rank 32 --client_num 10 --cloud_model_type large --edge_model_large_frac 0 --peft perTucker --cloud_peft Tlora --edge_alpha 0.3 --d2cloud_lr 0.005 --d2edge_lr 0.005 --startC 0 --endC 50
python main.py --aggregation doublecap --ModalityMixFrac 0 0.6 0.4 --iid 0.1 --rank 32 --client_num 10 --cloud_model_type large --edge_model_large_frac 1 --peft perTucker --cloud_peft Tlora --edge_alpha 0.3 --d2cloud_lr 0.005 --d2edge_lr 0.005 --startC 0 --endC 50

