# FedKD
The implementation of "FedKD: A Fine-Grained Parameter-Efficient Federated Co-tuning Framework with Knowledge Decoupling for Large and Small Foundation Models".  All simulations are conducted using the PyTorch framework on a server equipped with 8 NVIDIA GeForce RTX 4090 GPUs.
 

# Project Strcuture
```
	|-- data                    // download corresponding dataset and save here
	|-- requirements.txt    // the required environment on this dataset
	|-- run.bat    // running script of configurations in 6 heterogeneous settings on thsi dataset
	|-- mian.py    // main file of FedKD on this dataset
	|-- distill2cloud.py	// distillation to the cloud
	|-- distill2edge.py		// distillation to edge nodes
	|-- model.py 	// model configuration
	|-- options.py		// configuration for AsyncMMBF

```
# Environment
* Activate conda environment 
```
conda activate envname
```
* Enter the folder and install packages according to the corresponding requirements.txt
```
pip install -r requirements.txt
```

# Download datasets
* UTD-MHAD
    * UTD-MHAD dataset follows the previous work " <a href="https://github.com/xmouyang/Cosmo"> Cosmo: Contrastive Fusion Learning with Small Data for Multimodal Human Activity Recognition </a>". The pre-processed dataset and sample index can be downloaded in the [google driver folder](https://drive.google.com/file/d/1MOAiUdfo5JtCMt7AY7-n0hzCGhZ4bj_U/view?usp=drive_link).
    * Put the dataset into the UTD/data folder.
* nuScenes
    * The raw nuScenes dataset can be downloaded in the https://www.nuscenes.org/. The pre-processed dataset and sample index can downloaded in the [google driver folder](https://drive.google.com/file/d/16QucyiWzpzDUeHpB2JgOUv3nUcESA6Fz/view?usp=drive_link).
    * Put the dataset into the nuScenes/data folder.


# Quick Start 
* Activate the corresponding conda environment for dataset 
```
conda activate envname
```
* Run the script on your machine. 
```
run.bat
```
* For UTD-MHAD dataset, iid 0.1 and ModalityMixFrac 0 0.6 0.4 represent category-based and modality-based data heterogeneity, respectively; the model heterogeneity settings are set by controlling the edge_model_large_frac.
```
python main.py --aggregation doublecap --ModalityMixFrac 0 0 1 --iid 0.1 --rank 32 --client_num 10 --cloud_model_type large --edge_model_large_frac 1 --peft perTucker --cloud_peft Tlora --edge_alpha 0.3 --d2cloud_lr 0.005 --d2edge_lr 0.005 --startC 0 --endC 50
```
* For nuScenes dataset, iid 0.1 controls the category-based data heterogeneity; the model heterogeneity settings are set by the cloud_model_path and edge_model_path.
```
python main.py --cloud_model_path '/data/gm/DoubleCap/models/Qwen2-VL-2B-Instruct' --edge_model_path '/data/gm/DoubleCap/models/Qwen2-VL-2B-Instruct' '/data/gm/DoubleCap/models/Qwen2-VL-2B-Instruct' --aggregation doublecap --peft perTucker --cloud_peft Tlora --gcs 2 --edge_bs 2 2 2 --rank 32 --gpus 7 --iid 0.1 --epochs 2 --cloud_epochs 4 --participate_data_ratio 0.5 --communications 10 --edge_alpha 0.3 --lambda_lm 0.01 --edge_lr 0.05 0.05 --d2cloud_lr 0.05 --d2edge_lr 0.05
```
 
# Citation
The code and datasets of this project are made available for non-commercial, academic research only. If you find this work useful to you, please cite the following papers:
```  
@article{FedKD2025,
  title={FedKD: A Fine-Grained Parameter-Efficient Federated Co-tuning Framework with Knowledge Decoupling for Large and Small Foundation Models},
  author={},
  journal={},
  year={},
  publisher={}
}
```
