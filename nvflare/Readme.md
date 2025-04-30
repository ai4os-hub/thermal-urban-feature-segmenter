# Federated Learning with NVFlare for Thermal urban feature segmantation

This folder contains configuration files and setup instructions for training a **thermal-urban-feature-segmenter** model using **Federated Learning** with [NVFLARE](https://github.com/NVIDIA/NVFlare).

We provide setups to train the model using both the [**FedAvg**](https://arxiv.org/abs/1602.05629) and [**Swarm Learning**](https://www.nature.com/articles/s41586-021-03583-3) algorithms in a federated setting.

**NOTE:**
How to change an existing machine learning workflow into a federated learning workflow is shown [here](https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-world/ml-to-fl/tf#transform-cifar10-tensorflow-training-code-to-fl-with-nvflare-client-api).

---

##  How to Use

### 1. Clone the Repository and Install Dependencies

```bash
git clone -b dev https://codebase.helmholtz.cloud/m-team/ai/murnong_sem-segm-nvflare_public.git
cd murnong_sem-segm-nvflare_public/nvflare
pip install -r requirements_nvflare  
cd ..
pip install --ignore-installed blinker -e .

```

## ⚠️ Virtual Environment Recommendation

It is **strongly recommended** to use a virtual environment (e.g., `venv`) when setting up and running this project to avoid dependency conflicts.

---

## 2. MLFlow Configuration

This setup uses an **MLFlow instance** (from ai4eosc) to track experiments. To avoid errors and ensure compatibility, update the MLFlow tracking URI in the following file:

```bash
jobs/FedAvg/app/config/config_fed_server.conf
```

Specifically, modify the component with the ID:

```bash
mlflow_receiver_with_tracking_uri
```
in the last section of `components`.
For some decentralized workflows which do not include a server, the changes need to be applied within `config_fed_client.conf`. This applies for example to Swarm Learning.
When using a MLFlow server instance which is protected with a login, the credentials need to be exported within the terminal first. This can be done with:
```bash
export MLFLOW_TRACKING_USERNAME='your_username'
export MLFLOW_TRACKING_PASSWORD='your_password'
```

---

 

## 3. Running the Federated Training

###  Step 1: Activate the Virtual Environment
```bash
source /path/to/venv/bin/activate
```

###  Step 2: Run the NVFLARE Simulator

**For FedAvg:** Execte the following commnad in the terminal:
```bash
nvflare simulator -n 2 -t 2 ./jobs/FedAvg -w FedAvg_workspace

```


**For Swarm Learning:**

```bash
nvflare simulator -n 2 -t 2 ./jobs/Swarm_learning -w swarm_workspace
 

```

- `-n`: indicates the amount of sites within the project. This number should align with the number of sites within the config files and within the main code
- `-t`: indicates the amount of threads
- `-w`: indicates the path where the workspace should be created

# Structure of this project
```bash
├── jobs                            # Contains all federated learning job definitions for NVFLARE
│   ├── FedAvg                      # Job configuration for training using the FedAvg algorithm
│   │   ├── app                     # Application logic and components for FedAvg
│   │   │   ├── config              # Configuration files for FL clients and server
│   │   │   │   ├── config_fed_client.conf   # FedAvg client-side config (e.g., components, task settings)
│   │   │   │   └── config_fed_server.conf   # FedAvg server-side config (e.g., workflows, aggregators)
│   │   │   └── custom              # Custom Python modules used in training
│   │   │       ├── mlflow_receiver.py       # Logs training metrics to MLFlow via NVFLARE interface
│   │   │       ├── model_persistor.py       # Handles model saving/loading between training rounds
│   │   │       ├── split_data.py            # Splits dataset for each federated client
│   │   │       └── train_runfile_nvflare.py # Main training script executed by NVFLARE on each client
│   │   └── meta.conf              # Metadata file defining job roles, names, and dependencies
│   ├── split_data.py              # Optional script to globally pre-split datasets before job execution
│   └── Swarm                      # Job configuration for training using the Swarm Learning algorithm
│       ├── app                    # Application logic and components for Swarm
│       │   ├── config             # Configuration files for Swarm FL clients and server
│       │   │   ├── config_fed_client.conf   # Swarm client-side config
│       │   │   └── config_fed_server.conf   # Swarm server-side config
│       │   └── custom             # Custom training scripts and utilities for Swarm
│       │       ├── mlflow_receiver.py       # MLFlow integration for Swarm training
│       │       ├── model_persistor.py       # Model checkpoint handling
│       │       ├── split_data.py            # Data partitioning script (Swarm version)
│       │       └── train_runfile_nvflare.py # Main training script used in Swarm job
│       └── meta.conf             # Metadata for running the Swarm job in NVFLARE
├── README.md                     # Main project documentation and usage guide
└── requirements_nvflare.txt     # Required Python packages for running NVFLARE training jobs
     
```
 


---

##  More Information

For a complete guide on training the murnong_sem-segm-nvflare_public model and additional project details, please refer to the main `README.md` of the repository.



# References

Theoretical references:
 - Federated Learning: Collaborative Machine Learning without Centralized Training Data: https://blog.research.google/2017/04/federated-learning-collaborative.html
 - Roth, H. R., et al. (2022). NVIDIA FLARE: Federated Learning from Simulation to Real-World. arXiv. https://arxiv.org/abs/2210.13291
 - H. Brendan McMahan, et al. (2016). Communication-Efficient Learning of Deep Networks from Decentralized Data. arXiv. https://arxiv.org/abs/1602.05629

Technical references:
 - NVFlare GitHub Repository:  https://github.com/NVIDIA/NVFlare
 - NVFlare Documentation https://nvflare.readthedocs.io/en/2.5.0/

 ---