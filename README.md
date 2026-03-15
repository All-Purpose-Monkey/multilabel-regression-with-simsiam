# multilabel-regression-with-simsiam
End-to-end pipeline for creating a datase, simsiam training and prediction pipeline for UKDALE 2015 16kHz data. This is a showcase pipeline, taking excerpts of my master's thesis and goes over a baseline implementation for SimSiam architecture in STFT analysis of high fidelity data with the end goal of predicting aggregate and appliance power.   

Author: Yash Saraswat

email: yashsaraswat154154@gmail.com

# About the project

UK DALE 2015[https://dap.ceda.ac.uk/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2015/ReadMe_DALE.html] is a widely used dataset for Non Intrusive Load Metering (NILM) which deals with tasks relating to classification or regression of aggregate signal data into disaggregated signals. This project utilises the high fidelity data provided in the dataset and the individial appliance readings that act as ground truth for training a self supervised learning model. The choice of SimSiam archeticture[https://arxiv.org/abs/2011.10566] is to work with the weak labelling problem and higher robustness training with limited resources. Labelling ratio in the project dataset is improper as the individual appliance readings have many issues in them such as signal skips, interruptions and improper time jumps - creating a disaligned array - with Simsiam's pre-training regime you can utilize all stft images regardless of their labels. Secondly Simsiam archeticture is a low resource model which deals with robustness issues quite well - and has been optimized to run on an M1 CPU. This archeticure and its implementation was adapted from: https://medium.com/data-science/a-practical-guide-to-contrastive-learning-26e912c0362f

This project is licenced under CC-by-NC-SA 4.0, so in case you like my work, please feel free to get in touch with me if you see commercial usecases!

In terms of processes, this pipeline does the following to achieve the same:

## 1. Dataset Creation (functions defined in downloader.py and preprocess.py):
  1.1. Downloading 24hrs total of .flac files - stored to disk (4 GBs approx)
  
  1.2. Converting the 16kHz data to 6s long stft segments (using librosa and hann window) - 24*600 .npy files stored on disk
  
  1.3. Downloading main + appliance reading data
  
  1.4. Stitching the .dat files into an array and enforcing universal 6s window and mapping values
  
  1.5. Aligning windows by timestamps and creating a dataset to memory for training

## 2. Pre-Training using SimSiam architecture

   2.1. Train tests splits the data with a factor of 0.2 and stratification
   
   2.2. Employs encoder stategy using time frequency domain masking of stft segments - baseline 3 layer CNN using relu and batch normalisation 2D
   
   2.3. Encoder results are finetuned using a projection+prediction head and negative cosine loss in line with the archeticture to compute loss

## 3. Downstream Task - Multi-label power regression

   3.1. Drops all X,y rows from training and test without full labels (due to Nans caused by signal skips and interruptions) and normalizes y(reducing amplitude disturbances due to column wise difference) - out of 11520 training samples expect 4945 to survive
   
   3.2. Maps the backbone model and its vector representation output to a multi-label regression head
   
   3.3 15 epoch training regime followed with a MSE, MAE, RMSE, and R2 eval of the results by appliances and some plots to visualise the results.

# How to use:

You only need to run the simsiam_proto.py file, it has all the commands and function calls to make the pipeline run. Apart from that the functions are designed to be dynamic enough to handle increases or decreases to training data size with the help of easy to understand parameters (or so I hope) and comes with .yml file with the appropriate packages. Please run:

'conda activate ukdale_nilm'

on your terminal pefore running the pipeline.

# About Me
Hello! Born and raised in India, I am currently pursuing a masters in Data Science at Tilburg University, The Netherlands. I got interested in the field a few years back when I was working as marketing analyst for a fitness AI app startup which used computervision and transformers to give real-time feedback for home workouts. Since then I started learning to code, worked in more AI centered firms and now we are here :) My interests in data science (and the specialisation of my masters) is in deep learning for image, sound and sensor input data. Apart from that, I love to garden, cross-country biking and playing Dota 2. 
