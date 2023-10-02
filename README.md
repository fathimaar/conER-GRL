# conER-GRL
(Contextualized Emotion Recognition model using GNN with RL)

This repository contains all codes used for preprocessing, training and testing the conER-GRL model.

## Using the repository
Run the code files in the following order:
1. preprocess.py
2. domain_knowledge.py
3. ddqn_train.py

`python3 preprocess.py --path='data/IEMOCAP_features.pkl' --emotion_pair_size=5`

`python3 domain_knowledge.py --path='data/IEMOCAP_features.pkl' --emotion_pair_size=5`

`python3 ddqn_training.py`
