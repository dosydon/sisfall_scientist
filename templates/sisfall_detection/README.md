# Fall Detection Using AI Scientist

This template contains codes for fall detection using [AI Scientist v1](https://github.com/SakanaAI/AI-Scientist). Falls represent a major health risk for elderly individuals. In today's rapidly aging society, automated fall detection systems are becoming increasingly crucial for ensuring safety and providing timely medical assistance. The provided code uses the [Sisfall dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC5298771/) to train LSTM from  accelerometer data from wearable sensors.

## Installation
```
pip install -r requirements.txt
```

## Data Preparation
Download [Sisfall dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC5298771/) and [additional annotations](https://arxiv.org/abs/1804.04976). Use 'preprocess.py' to preprocess the data and place it under 'data/sisfall'.

## Running
```
python launch_scientist.py --model "claude-3-5-sonnet-20241022"  --experiment sisfall_detection 
```