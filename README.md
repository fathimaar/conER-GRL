# conER-GRL
(Contextualized Emotion Recognition model using GNN with RL)

This repository contains all codes used for preprocessing, training and testing the conER-GRL model.

## Abstract
With the development of generative artificial intelligence (AI) and language models, conversational agents are gaining acceptance in society. They provide a human touch to the user by interacting in a way that is familiar to them. As a result, they are being used more as a companion than as information-seeking tools. It is therefore important to understand the user’s emotions in order to provide considerate responses. Compared to the standard emotion recognition problem, conversational agents face an additional constraint that the recognition should be done in real time. Studies on model architectures that use audio, visual and textual modalities have mainly focused on classifying emotions using full video sequences that do not provide online capabilities. Such methods can predict emotions only after the videos have been completed which limits their usefulness in practical scenarios such as conversational agents. In this work, we present a novel paradigm for contextualized Emotion Recognition using Graph Neural Network with Reinforcement Learning (conER-GRL). The system proposes Gated Recurrent Units (GRU) to extract features from conversations. GRUs are partitioned into smaller groups of utterances to enable effective acquisition of contextual information. To capture the complex dependencies of emotion features in interactive scenarios, Graph Neural Networks (GNN) and Reinforcement Learning (RL) agents are trained in a cascaded fashion. Comparing the results from the conER-GRL model with other state-of-the-art models on the publicly available IEMOCAP dataset, demonstrates the advantageous capabilities of the conER-GRL architecture.
[Key words: Multi-modal Emotion Recognition, Graph Neural Network, Deep Reinforcement Learning]

## Using the repository
Run the code files in the following order:
1. preprocess.py
2. dataloader.py
3. domain_knowledge.py
4. ddqn_train.py
5. dialogue_level_test.py
6. pair_dataloader.py


