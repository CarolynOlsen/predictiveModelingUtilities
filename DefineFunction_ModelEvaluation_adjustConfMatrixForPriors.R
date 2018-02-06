
# PURPOSE: This model takes a classification model's
# confusion matrix as input, when the model was trained
# on an artificially balanced data set. It returns 
# model performance metrics, adjusted for prior probabilities.

adjustForPriors <- function(confmatrix, eventrate) {
  
  tn <- confmatrix$table[1,1]
  fn <- confmatrix$table[1,2]
  fp <- confmatrix$table[2,1]
  tp <- confmatrix$table[2,2]
  
  
  sens <- tp / (tp + fn)
    
  spec <- tn / (tn + fp)
  
  ppv <- (sens * eventrate) / ((sens * eventrate) + (1-eventrate)*(1-spec) )
  
  npv <- (spec*(1-eventrate)) / ( spec*(1-eventrate) / (spec*(1-eventrate) + (1-sens)*eventrate  ) )
  
  acc <- (tp + tn) / (tp + tn + fp + fn)
  
  print(paste0("Accuracy = ",round(acc,3)))
  print(paste0("Sensitivity = ",round(sens,3)))
  print(paste0("Specificity = ",round(spec,3)))
  print(paste0("PPV = ",round(ppv,3)))
  print(paste0("NPV = ",round(npv,3)))
  
}