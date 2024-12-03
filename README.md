# Group7_Cambricon-D
This is a project about Implementation of Cambricon-D using for accelerating hardware architecture design for diffusion models, utilizing differential computation to shrink redundant calculations and address memory access.
Thanks for SCALE Sim [SCALE Sim](https://github.com/scalesim-project/scale-sim-v2) provides a simulator for systolic array based accelerators for Convolution, Feed Forward, and any layer that uses GEMMs.

## Introduction
Our goal is to accelerate hardware architecture design for diffusion models, utilizing differential computation to shrink redundant calculations and address memory access. By focusing on differential values, or "deltas," between iterations rather than recalculating the entire dataset, it is possible to reduce both the computational and memory access costs, aiming to increase processing efficiency while maintaining the model's performance accuracy.

## Sturcture
* ./result     //All the experiment results and diagram we generated
* ./scale-sim-v2    //Scale-Sim
    - ./code-example
    - ./configs    //Store configurations
    - ./scalesim    //Main simulator
    - ./topologies    //Store types of topology
    - Makefile
    - requirement.txt
    - setup.py
* ./test_run    //Store the execution results

## Install steps
Clone this repository using 
```bash
$ git clone https://github.com/CSCE-614-Dr-Kim-Fall-2024/Group7_Cambricon-D.git
$ cd Group7_Cambricon-D
$ cd scale-sim-v2
```

To make sure those file have access permissions, put Group7_Cambricon-D folder inside any folder.
After entering scale-sim-v2, you can run scalesim/scale.py to start your simulation. We have cambricon_d.py in our main folder but it still in progressing. In this process the configuration architecture provided in /config and topology descriptor provided in /topologies are necessary. In order to access those functions:

```bash
$ python scalesim/scale.py -c <path_to_config_file> -t <path_to_topology_file> 
```

For instance:

```bash
$ python scalesim/scale.py -c configs/A100.cfg -t topologies/guid_diffusion/GUID128.csv
```
or  
```bash
$ python scalesim/scale.py -c configs/A100.cfg -t topologies/conv_nets/alexnet.csv
```

## Configuration
Configurations are stored in /config including scale.cfg, and A100.cfg. We use NVIDIA A100 as our specific accelerator architecture. The baseline data is following by:

![image](https://github.com/user-attachments/assets/769af500-62d7-4fc6-8f69-1b19d919fa34)


## Tolologies
For other layer types, SCALE-Sim also accepts workload descriptions in a specific format. For the guided-diffusion model, such as 128x128, we assume the model consists of 3 convolutional layers and 3 deconvolutional layers. Each layer is characterized by the following parameters:

![image](https://github.com/user-attachments/assets/bf7b97f7-d244-40ea-955d-b25dc006730f)


## Output  
After executing the command, the results are stored in the /test_run directory. The folder name is specified in the configuration file, where it is defined as run_name = \<name\>.

![image](https://github.com/user-attachments/assets/de8d9814-1744-47b9-b6ff-d6737315b0b3)  

For example, if \<name\> is equal to  NVIDIA_A100_simulation, then all the executed information would be stored in test_run/NVIDIA_A100_simulation with below three csv files:  
BANDWIDTH_REPORT.csv  
COMPUTE_REPORT.csv  
DETAILED_ACCESS_REPORT.csv 

When we execute the commands every time the previous results will be replaced. So we maintained the results in GUID_XXX folder.
  
## Team Members
Emily Chang(334002383)  
Chi-Wei Ho(335006925)  
Yu-Ting Lin(535005629)  

## Reference
[1] W. Kong, Y. Hao, Q. Guo, Y. Zhao, X. Song, X. Li, M. Zou, Z. Du, R. Zhang, C. Liu, Y. Wen, P. Jin, X. Hu, W. Li, Z. Xu, and T. Chen, “Cambricon-d: Full-network differential acceleration for diffusion models,” in ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA), 2024. [paper](https://www.computer.org/csdl/proceedings/isca/2024/1Z3pw66W6DC)  
[2] A. Samajdar, J. M. Joseph, Y. Zhu, P. Whatmough, M. Mattina, and T. Krishna, “A systematic methodology for characterizing scalability of dnn accelerators using scale-sim,” in 2020 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS). IEEE, 2020, pp. 58–68. [paper](https://arxiv.org/abs/1811.02883)
