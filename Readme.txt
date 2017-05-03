# CSC591
# Team: S9
# A Sepsis-Shock Prediction Model Using Hidden Markov Model

0. Environment: Python 3.5 with the package hmmlearn

1. Files description
	Run: python <file_name>.py
   1) RevKM_supervised.py: Supervised HMM with k-means(KM) for new observations
   
   2) RevKM_semisupervised.py: Semi-supervised HMM with k-means(KM) for new observations
   
   3) RevDP_supervised.py£ºSupervised HMM with Dirichelet Process(DP) for new observations
   
   4) RevDP_semisupervised.py: Semi-supervised HMM with Dirichelet Process(DP) for new observations
   
2. Early prediction
    Run: python sepsis_earlypred.py
	- default mode: diagnosis hmm model
       - To change the mode: single/multi hmm for early prediction, 
         edit the if-statement for those codes from False to True.