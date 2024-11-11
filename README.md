# Introduction

This is the implementation of our paper *[FedALA: Adaptive Local Aggregation for Personalized Federated Learning](https://ojs.aaai.org/index.php/AAAI/article/view/26330)* (accepted by AAAI 2023). An extended version (derivation of Equation (6), hyperparameter settings, etc.) can be found at https://arxiv.org/pdf/2212.01197v4.pdf.

- [Oral PPT](./FedALAOral.pdf)
- [Poster PDF](./FedALAPoster.pdf)


## Citation

```
@inproceedings{zhang2023fedala,
  title={Fedala: Adaptive local aggregation for personalized federated learning},
  author={Zhang, Jianqing and Hua, Yang and Wang, Hao and Song, Tao and Xue, Zhengui and Ma, Ruhui and Guan, Haibing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={9},
  pages={11237--11244},
  year={2023}
}
```


# Datasets and Environments

Here, we only upload the mnist dataset in the default heterogeneous setting with Dir(0.1) for example. You can generate other datasets and environment settings following [PFLlib](https://github.com/TsingZ0/PFLlib).


# System

- `main.py`: configurations of **FedALA**. 
- `run_me.sh`: start **FedALA**. 
- `env_linux.yaml`: python environment to run **FedALA** on Linux. 
- `./flcore`: 
    - `./clients/clientALA.py`: the code on the client. 
    - `./servers/serverALA.py`: the code on the server. 
    - `./trainmodel/models.py`: the code for backbones. 
- `./utils`:
    - `ALA.py`: the code of our **Adaptive Local Aggregation (ALA)** module
    - `data_utils.py`: the code to read the dataset. 

# Adaptive Local Aggregation (ALA) module

`./system/utils/ALA.py` is the implementation of the ALA module, which corresponds to the pseudocode from `line 6` to `line 16` in Algorithm 1 in our paper. You can easily apply the ALA module to other federated learning (FL) methods by importing it as a Python module. 

## Details
- `adaptive_local_aggregation`: It prepares the work before weight learning, randomly selecting local training data and preserving the lower layers of the update. Since the parameters in the models will be changed during the ALA process, we use their references for convenience. Then, it learns the weight for local aggregation. Firstly, to prevent influencing the local model training, we clone the local model as the *temp local model*, only for obtaining the gradients after backpropagation of the local model during the weight learning. Then, we freeze the parameters of the lower layers of the *temp local model* to prevent gradients computation in Pytorch. We initialize the weight and *temp local model* before weight training. After that, we train the weight until convergence in the second iteration and train only one epoch in the subsequent iterations. Finally, we set the parameters of the *temp local model* to the corresponding parameters of the local model to obtain the initialized local model. 

## Illustrations

- Local learning process on client i in the t-th iteration. Specifically, client i downloads the global model from the server, locally aggregates it with the old local model by ALA module for local initialization, trains the local model, and finally uploads the trained local model to the server.

![](./figs/illustrate.jpg)

- The learning process in ALA. LA denotes "local aggregation". Here, we consider a five-layer model and set p=3. The lighter the color, the larger the value.

![](./figs/ALA.jpg)

- Update direction correction. 2D visualization of local learning trajectory (from iteration 140 to 200) and the local loss surface on MNIST in the pathological heterogeneous setting. The green square dots and red circles represent the local model at the beginning and end of each iteration, respectively. The black and blue trajectories with the arrows represent FedAvg and FedALA, respectively. The local models are projected to the 2D plane using PCA. C1 and C2 are the two principal components generated by the PCA.

![](./figs/correction.png)

## How to use
- Step 1: Import the ALA module as a Python module.
```
import ALA
```

- Step 2: Then, feed the required parameters (please refer to `ALA.py` for more details) to initialize the `ALA` module.
```
class Client(object):
    def __init__(self, ...):
        # other code
        self.ALA = ALA(self.id, self.loss, self.train_data, self.batch_size, 
                    self.rand_percent, self.layer_idx, self.eta, self.device)
        # other code
```

- Step 3: Thirdly, feed the reveived global model and the old local model to `self.ALA.adaptive_local_aggregation()` for local initialization. 
```
class Client(object):
    def __init__(self, ...):
        # other code
        self.ALA = ALA(self.id, self.loss, self.train_data, self.batch_size, 
                    self.rand_percent, self.layer_idx, self.eta, self.device)
        # other code

    def local_initialization(self, received_global_model, ...):
        # other code
        self.ALA.adaptive_local_aggregation(received_global_model, self.model)
        # other code
```

# Training and Evaluation

All codes corresponding to **FedALA** are stored in `./system`. Just run the following commands.

```
cd ./system
sh run_me.sh
```

**Note**: Due to the dynamics of the *floating-point calculation accuracy* of different GPUs, you may need to set a suitable `threshold` (we set it to 0.01 in our paper by default) for the ALA module to control its convergence level in the start phase. A small `threshold` may cause your system to get *stuck* in the first iteration.
