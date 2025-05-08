# Federated Learning with NVFlare for Thermal urban feature segmantation

In this work we applied Federated Learning on a use case of Thermal urban feature semantic segmentation. Unet was used to identify thermal anomalies (hot spots) in urban environments. 
This, to improve the efficiency of energy-related systems.
Due to multiple cities involved and the growing attention in regard of privacy, introducing Federated Learning seemed like a common step.
In particular, [NVFlare](https://github.com/NVIDIA/NVFlare) and its variaty of workflows and features was used.

**NOTE:**
How to change an existing machine learning workflow into a federated learning workflow is shown [here](https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-world/ml-to-fl/tf#transform-cifar10-tensorflow-training-code-to-fl-with-nvflare-client-api).

---

##  How to Use

### 1. Clone the Repository and Install Dependencies

```bash
git clone -b nvflare https://github.com/ai4os-hub/thermal-urban-feature-segmenter.git # Clone the 'nvflare' branch of the repository
cd thermal-urban-feature-segmenter/TUFSeg   
git init submodule # Initialize submodules in the cloned repo
git submodule update --remote --merge # Update submodules to the latest commit from their remote and merge changes   
pip install -e . # Install TUFSeg package
cd ..
pip install -e . # Install the deepaas api requirements
cd nvflare
pip install -r requirements_nvflare # Install the NVFlare requirements
```

## ⚠️ Virtual Environment Recommendation
> **Note:** It is **strongly recommended** to either use a virtual environment (e.g., `venv`) or the [Dockerfile](/dockerfile) provided to build the Docker image when setting up and running this project, in order to avoid dependency issues.

---

## 2. MLFlow Configuration

This setup uses an **MLFlow instance** (from ai4eosc) to track experiments. To avoid errors and ensure compatibility, update the MLFlow tracking URI in the following file:

```bash
jobs/$[Job_name]/app/config/config_fed_server.conf
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

Execute the following commnad in the terminal:
```bash
nvflare simulator -n 2 -t 2 ./jobs/$[Job_name] -w path/to/workspace

```


- `-n`: indicates the amount of sites within the project. This number should align with the number of sites within the config files and within the main code
- `-t`: indicates the amount of threads
- `-w`: indicates the path where the workspace should be created

# Structure of this project
```bash
├── jobs                            # Contains all federated learning job definitions for NVFLARE
│   ├── Job_name                    # Job configuration for training
│   │   ├── app                     # Application logic and components for FedAvg
│   │   │   ├── config              # Configuration files for FL clients and server
│   │   │   │   ├── config_fed_client.conf   # Client-side config (e.g., components, task settings)
│   │   │   │   └── config_fed_server.conf   # Server-side config (e.g., workflows, aggregators)
│   │   │   └── custom              # Custom Python modules used in training
│   │   │       ├── evaluate.py              # Evaluation script for evaluating the trained model
│   │   │       ├── mlflow_receiver.py       # Logs training metrics to MLFlow via NVFLARE interface
│   │   │       ├── model_persistor.py       # Handles model saving/loading between training rounds
│   │   │       └── train_runfile_nvflare.py # Main training script executed by NVFLARE on each client
│   │   └── meta.conf              # Metadata file defining job roles, names, and dependencies
│   ├── Another_Job                # Another Job configuration
│   └── ...                        # More Job configurations
├── jobs_with_tracking             # Job configurations with eneabled energy consumption tracking using perun
├── README.md                      # Main project documentation and usage guide
└── requirements_nvflare.txt       # Required Python packages for running NVFLARE training jobs
     
```
 


---

##  More Information

For a complete guide on training the thermal-urban-feature-segmenter model and additional project details, please refer to the main [`README.md`](../README.md) of the repository.



# References

Theoretical references:
 - [Federated Learning] Collaborative Machine Learning without Centralized Training Data: https://blog.research.google/2017/04/federated-learning-collaborative.html
 - [NVFlare] Roth, H. R., et al. (2022). NVIDIA FLARE: Federated Learning from Simulation to Real-World. arXiv. https://arxiv.org/abs/2210.13291
 - [FedAvg] H. Brendan McMahan, et al. (2016). Communication-Efficient Learning of Deep Networks from Decentralized Data. arXiv. https://arxiv.org/abs/1602.05629
 - [FedProx] Li, T., Sahu, A.K., Zaheer, M., Sanjabi, M., Talwalkar, A., Smith, V. (2020): Federated optimization in heterogeneous networks, 
https://arxiv.org/abs/1812.06127
 - [FedOpt] Sashank J. Reddi and Zachary Charles and Manzil Zaheer and Zachary Garrett and Keith Rush and Jakub Kone (2021), Adaptive Federated Optimization: https://arxiv.org/abs/2003.00295
 - [Scaffold] Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank J. Reddi, Sebastian U. Stich, Ananda Theertha Suresh (2021): SCAFFOLD: Stochastic Controlled Averaging for Federated Learning: https://arxiv.org/abs/1910.06378
 - [Swarm Learning] Warnat-Herresthal, Stefanie and Schultze, Hartmut and Shastry, Krishnaprasad Lingadahalli et al. (2021), Swarm Learning for decentralized and confidential clinical machine learning: https://www.nature.com/articles/s41586-021-03583-3
 - [Cyclic Learning] Chang, Ken and Balachandar, Niranjan and Lam, Carson and Yi et al. (2018), Distributed deep learning networks among institutions for medical imaging: https://pubmed.ncbi.nlm.nih.gov/29617797/
 - [Perun] Gutiérrez Hermosillo Muriedas, J.P., Flügel, K., Debus, C., Obermaier, H., Streit, A., Götz, M.: perun: Benchmarking Energy Consumption of High-Performance Computing Applications. Euro-Par 2023: Parallel Processing. pp. 17–31. Springer Nature Switzerland, Cham (2023). https://doi.org/10.1007/978-3-031-39698-4_2.


Technical references:
 - NVFlare GitHub Repository:  https://github.com/NVIDIA/NVFlare
 - NVFlare Documentation https://nvflare.readthedocs.io/en/2.5.0/

 ---
