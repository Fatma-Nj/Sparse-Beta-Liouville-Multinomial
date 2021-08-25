# Sparse Document Analysis using Beta-Liouville Naive Bayes with Vocabulary Knowledge 

Fatma Najar and Nizar Bouguila  

Published in 16th International Conference on Document Analysis and Recognition, ICDAR 2021


DATA FILES
============================================================
- Emotion_data
- hatespeech_data

This file contains all the text datasets used in this article


ALGORITHMS FILES
===========================================================
- beta_liouville_bayesian.py

*the python code that contains the implementation of the sparse Beta-liouville naive bayes
The splitting of training/testing set is randomly affected that's why we run and average over 10 times the algorihtm 
Each run gives a slight different results.

- sparse_multinomial.py

*the python code that contains the related-works of DS-VK, DS, and MNB (mentioned in the paper).
