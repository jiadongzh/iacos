# iACOS: Advancing Implicit Sentiment Extraction with Informative and Adaptive Negative Examples (NAACL 2024)

## Requirements

Run the following commands to install requirements:

Note that: select a cuda version for torch based on your own GPU.

```setup
(1) conda create -n iacos python==3.7.0
(2) conda activate iacos
(3) pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
(4) pip install transformers==4.25.1
(5) pip install pandas==1.1.5
(6) pip install tensorboard==2.11.2
(7) pip install pytorch-crf==0.7.2
(8) pip install urllib3==1.26.15
```

## Training
Note that:

(1) For each command, we only need to set partial hyperparameters, the other ones are automatically set to default values.

(2) After each epoch, we evaluate the current model on validation and testing data, and output the results with columns:

epoch: current epoch number

train_loss: train loss per batch

dev_pre: Precision on validation

dev_rec: Recall on validation

dev_f1: F1 on validation

precision: on test data

recall: on test data

f1: on test data

f1-00: on test data with explicit aspects and explicit opinions

f1-01: on test data with explicit aspects and implicit opinions

f1-10: on test data with implicit aspects and explicit opinions

f1-11: on test data with implicit aspects and implicit opinions

(3) These results of each command will be saved at ./runs/path_with_hyperparameter_values/metrics.csv which will be used for Evaluation.


### 1. Convergence analysis 

Run the following commands with any given random seed on two datasets, respectively.
```
python train.py -seed 0 -data laptop
python train.py -seed 0 -data rest16 
```
We need to run the same command with 10 different random seeds to obtain standard deviation of metrics.

### 2. Effect of negative sampling methods
2.1 Run the following commands for "Random" negative sampling on two datasets, respectively (4096 denotes the maximum number of negative samples in each batch data to avoid out-of-memory).
```
python train.py -seed 0 -data laptop -ng -4096
python train.py -seed 0 -data rest16 -ng -4096
```

2.2 Run the following commands for "None" negative sampling on two datasets, respectively.
```
python train.py -seed 0 -data laptop -ng 0
python train.py -seed 0 -data rest16 -ng 0 
```

### 3. Ablation experiment
3.1 Run the following commands without multi-head attention on two datasets, respectively.
```
python train.py -seed 0 -data laptop -mh 0
python train.py -seed 0 -data rest16 -mh 0
```

3.2 Run the following commands without multi-task learning on two datasets, respectively.
```
python train.py -seed 0 -data laptop -mt ''
python train.py -seed 0 -data rest16 -mt '' 
```

3.3 Run the following commands without implicit tokens on two datasets, respectively.
```
python train.py -seed 0 -data laptop -imp '' -cls 1
python train.py -seed 0 -data rest16 -imp '' -cls 1
```

## Evaluation

Note that:

(1) The train.py have evaluated the model on validation and testing data and saved the results at ./runs/path_with_hyperparameter_values/metrics.csv

(2) As mentioned in convergence analysis of the paper, 500 epochs are equally divided into five bins and the mean performance is calculated for each bin, along with standard deviation.

(3) In the paper, unless otherwise specified, we report results at 400 epochs.

Run the following command to get the results on five bins of 500 epochs in a trial: 
```eval
python eval.py -csv ./runs/path_with_hyperparameter_values/metrics.csv
```
Output Examples:

epoch,precision_mean,precision_std,recall_mean,recall_std,f1_mean,f1_std,f1-00_mean,f1-10_mean,f1-01_mean,f1-11_mean
100,0.4941,0.2251,0.2760,0.1753,0.3276,0.1910,0.3788,0.2113,0.1157,0.2670
200,0.5606,0.0158,0.4864,0.0192,0.5205,0.0122,0.5886,0.4108,0.2181,0.4088
300,0.5752,0.0179,0.5199,0.0134,0.5459,0.0109,0.6096,0.4589,0.2473,0.4421
400,0.5724,0.0134,0.5321,0.0089,0.5514,0.0074,0.6166,0.4778,0.2370,0.4345
500,0.5573,0.0134,0.5311,0.0075,0.5438,0.0076,0.6057,0.4525,0.2491,0.4563

Run the above command for different random trials to get the mean and std at 400 epochs.
