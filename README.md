preprocessing : contains all the .py files useful for handling data and preprocessing it before the ML part

machine learning : contains the files creating the machine learning models and implementing the training and testing of the models

visualisation : create graphs and results visualisation in this part

main.py : file implementing the whole pipeline


The overall aim is to use these data recorded from the last few years in our ICU to come up with predictive models for a variety of end points
  Organ failure, death, length of ICU stay and two really useful parameters - predicting the time that we can take people off mechanical ventilation and the time we can stop giving people drugs to support their blood pressure.
  
  
  For now I've used random forest & gradient boosting to try predict this thing FiCOFa
    which is a combined score of different organ failures and the requirements for organ support at any one time
    
    
   Its currently a bit rubbish, giving r2 values of 0.65
   
   Steps that will improve things
   
   1] Plug this in to a DNN [?LSTM] using Keras
   2] Split each patient out individually into a 48hr observation period with associated 24hr prediction window
   3] Including rolling & expanding means to describe trends in different features since:
          a) the beginning of the observation window
          b) admission to ICU
   4] A more nuanced way of imputing missing values
