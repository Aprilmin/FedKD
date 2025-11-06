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
    * UTD-MHAD dataset follows the previous work " <a href="https://github.com/xmouyang/Cosmo"> Cosmo: Contrastive Fusion Learning with Small Data for Multimodal Human Activity Recognition </a>" and pre-processed dataset can be downloaded in the [google driver folder]().
    * Put the dataset into the UTD/data folder.
* HA4M
    * The raw nuScenes dataset can be downloaded in the https://www.nuscenes.org/. The pre-processed dataset and sample index can downloaded in the [google driver folder]().
    * Put the dataset into the nuScenes/data folder.


# Quick Start 
* Activate the corresponding conda environment for dataset 
```
conda activate envname
```
* Run the script on your machine. Note that iid=0.1, ModalityMixFrac=0 0.6 0.4 represent category-based and modality-based data heterogeneity respectively, vice versa. For UTD-MHAD dataset, the model heterogeneity settings are set by controlling the edge_model_large_frac; For nuScenes dataset, the model heterogeneity settings are set by the cloud_model_path and edge_model_path.
```
run.bat
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
