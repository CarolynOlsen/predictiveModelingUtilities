
# PURPOSE: Bootstrap Aggregated ensemble modeling, where 
# the output includes Out of Bag (OOB) sample performance. 
# Flexible split criteria, either Gini, Information, or 
# both. 

bagging <- function(y_name
                    ,training_df
                    ,scoring_df
                    ,splitcriteria="gini"
                    ,usepriors=TRUE
                    ,iterations=30
                    ,starting_seed=12345
                    ,complexityparameter=0.01
                    ,minbucket=5
                    ,maxdepth=10) {
  
  library(dplyr)
  library(caret)
  library(rpart)
  
  #get prior probabilities
  prior_positive <- table(training_df[,y_name])[2]/(table(training_df[,y_name])[1]+table(training_df[,y_name])[2])
  
  #create empty df for performance statistics
  oob_results <- as.data.frame(matrix(ncol=22,nrow=iterations))
  colnames(oob_results) <- c("Iteration","Cutoff"
                             ,"Train_TruePos","Train_FalsePos","Train_TrueNeg","Train_FalseNeg"
                             ,"Train_Acc","Train_Sensitivity","Train_Specificity","Train_BalancedAcc"
                             ,"OOB_pct","OOB_TruePos","OOB_FalsePos","OOB_TrueNeg","OOB_FalseNeg"
                             ,"OOB_Acc","OOB_Kappa","OOB_Sensitivity","OOB_Specificity","OOB_Prevalence","OOB_DetectionRate","OOB_BalancedAcc")
  
  #create empty df for predictions
  bagged_preds <- as.data.frame(matrix(ncol=iterations,nrow=nrow(scoring_df)))
  for (x in 1:iterations) {
    if (x%%2==0) {
      colnames(bagged_preds)[x] <- paste0("pred",x,"_gini")
    } else {
      colnames(bagged_preds)[x] <- paste0("pred",x,"_info")
    }
  }
  
  #create empty df for variable importance
  varimp <- as.data.frame(matrix(ncol=iterations+1,nrow=ncol(training_df)-1))
  colnames(varimp)[1] <- "varimp_variable"
  for (x in 1:iterations) {
    if (x%%2==0) {
      colnames(varimp)[x+1] <- paste0("varimp",x,"_gini")
    } else {
      colnames(varimp)[x+1] <- paste0("varimp",x,"_info")
    }
  }
  varimp$varimp_variable <- colnames(training_df)[!(colnames(training_df)==y_name)]
  varimp <- varimp[order(varimp$varimp_variable),]
  
  #define potential cutoff values, step by 0.01
  potential_cutoffs <- seq(from=0.01,to=0.99,by=0.01)
  
  #define series of random seeds
  seeds <- seq(from=starting_seed,to=(starting_seed+iterations),by=1)
  
  for (i in 1:iterations) {
    
    # create bootstrap sample
    # randomly sample df with replacement
    set.seed(seeds[i])
    sample_df <- sample_n(training_df,nrow(training_df),replace=TRUE)
    
    # identify out of bag holdout
    oob <- training_df[!(rownames(training_df) %in% rownames(sample_df)),]
    
    #estimate decision trees
    #alternate between splitting criteria
    if (splitcriteria == "both") {
      if (i%%2==0) {
        set.seed(seeds[i])
        #splits based on gini index
        model <- rpart(sample_df[,y_name] ~ .
                       ,data=sample_df[,!names(sample_df) %in% c(y_name)]
                       ,method="class"
                       # ,weights=(as.integer(sample_df[,y_name])*25)
                       ,parms=list(prior=c(1-prior_positive,prior_positive),split="gini")
                       ,control=rpart.control(minbucket=minbucket
                                              ,cp=complexityparameter
                                              ,maxdepth=maxdepth))
      } else {
        set.seed(seeds[i])
        #splits based on information gain
        model <- rpart(sample_df[,y_name] ~ .
                       ,data=sample_df[,!names(sample_df) %in% c(y_name)]
                       ,method="class"
                       # ,weights=(as.integer(sample_df[,y_name])*25)
                       ,parms=list(prior=c(1-prior_positive,prior_positive), split="information")
                       ,control=rpart.control(minbucket=minbucket
                                              ,cp=complexityparameter
                                              ,maxdepth=maxdepth))
      }
    } else if (splitcriteria == "gini") {
      set.seed(seeds[i])
      #splits based on gini index
      model <- rpart(sample_df[,y_name] ~ .
                     ,data=sample_df[,!names(sample_df) %in% c(y_name)]
                     ,method="class"
                     # ,weights=(as.integer(sample_df[,y_name])*25)
                     ,parms=list(prior=c(1-prior_positive,prior_positive),split="gini")
                     ,control=rpart.control(minbucket=minbucket
                                            ,cp=complexityparameter
                                            ,maxdepth=maxdepth))
    } else if (splitcriteria == "information") {
      set.seed(seeds[i])
      #splits based on information gain
      model <- rpart(sample_df[,y_name] ~ .
                     ,data=sample_df[,!names(sample_df) %in% c(y_name)]
                     ,method="class"
                     # ,weights=(as.integer(sample_df[,y_name])*25)
                     ,parms=list(prior=c(1-prior_positive,prior_positive), split="information")
                     ,control=rpart.control(minbucket=minbucket
                                            ,cp=complexityparameter
                                            ,maxdepth=maxdepth
                                            ,minsplit=0))
    }
    
    
    #save variable importance
    model_varimp <- as.data.frame(varImp(model))
    model_varimp$Variable <- row.names(model_varimp)
    model_varimp <- model_varimp[order(model_varimp$Variable),]
    varimp[,i+1] <- model_varimp[,1]
    print(varImp(model))
    
    #predict training probabilities
    set.seed(seeds[i])
    pred_y_prob <- as.data.frame(predict(model,sample_df,type="prob"))$`1`
    
    #define predicted values for training sample and OOB sample
    if (usepriors==TRUE) {
      pred_y_train <- as.factor(ifelse(pred_y_prob>=prior_positive,1,0))
      levels(pred_y_train) <- c("0","1")
      pred_y_oob <- as.factor(ifelse(as.data.frame(predict(model,oob,type="prob"))$`1`>=prior_positive,1,0))
      levels(pred_y_oob) <- c("0","1")
      cutoff <- prior_positive
    } else {
      pred_y_train <- as.factor(ifelse(pred_y_prob>=.5,1,0))
      levels(pred_y_train) <- c("0","1")
      pred_y_oob <- as.factor(ifelse(as.data.frame(predict(model,oob,type="prob"))$`1`>=.5,1,0))
      levels(pred_y_oob) <- c("0","1")
      cutoff <- 0.50
    }
    
    #capture train performance statistics
    result_train <- confusionMatrix(pred_y_train
                                    ,sample_df[[y_name]]
                                    ,positive="1")
    result_list_train <- as.list(c("TruePos"=result_train$table[2,2]
                                   ,"FalsePos"=result_train$table[2,1]
                                   ,"TrueNeg"=result_train$table[1,1]
                                   ,"FalseNeg"=result_train$table[1,2]
                                   ,result_train$overall["Accuracy"]
                                   ,result_train$byClass[c("Sensitivity","Specificity","Balanced Accuracy")]))
    names(result_list_train) <- names(oob_results[,grepl("Train",names(oob_results))])
    
    #test statistics using out of bag sample
    result_oob <- confusionMatrix(pred_y_oob
                                  ,oob[[y_name]]
                                  ,positive="1")
    result_list_oob <- as.list(c("OOB_pct"=nrow(oob)/nrow(training_df)
                                 ,"TruePos"=result_oob$table[2,2]
                                 ,"FalsePos"=result_oob$table[2,1]
                                 ,"TrueNeg"=result_oob$table[1,1]
                                 ,"FalseNeg"=result_oob$table[1,2]
                                 ,result_oob$overall[c("Accuracy","Kappa")]
                                 ,result_oob$byClass[c("Sensitivity","Specificity","Prevalence","Detection Rate","Balanced Accuracy")]))
    names(result_list_oob) <- names(oob_results[,grepl("OOB",names(oob_results))])
    
    #combine result lists
    result_list <- c("Iteration"=i,"Cutoff"=cutoff,result_list_train,result_list_oob)
    
    #add the iteration's results to the data frame of out-of-bag results
    oob_results[i,] <- result_list
    
    if (splitcriteria=="both") {
      if (i%%2==0) {
        print(paste0("Iteration ",i," complete. OOB balanced accuracy ",round(result_list_oob$OOB_BalancedAcc*100,1),"%. Split = Gini."))
      } else {
        print(paste0("Iteration ",i," complete. OOB balanced accuracy ",round(result_list_oob$OOB_BalancedAcc*100,1),"%. Split = Information Gain."))
      }
    } else if (splitcriteria=="gini") {
      print(paste0("Iteration ",i," complete. OOB balanced accuracy ",round(result_list_oob$OOB_BalancedAcc*100,1),"%. Split = Gini."))
    } else if (splitcriteria=="information") {
      print(paste0("Iteration ",i," complete. OOB balanced accuracy ",round(result_list_oob$OOB_BalancedAcc*100,1),"%. Split = Information Gain."))
    }
    
    
    #define predicted values for scoring data set
    set.seed(seeds[i])
    pred_y_prob_full <- as.data.frame(predict(model,newdata=scoring_df,type="prob",verbose=TRUE))$`1`
    #add the predicted probabilties to the predictions data frame
    bagged_preds[,i] <- pred_y_prob_full
   
  }
  
  #create average importance field
  varimp$varimp_avg <- apply(varimp[,-1],1,function(x) mean(x))
  
  #identify whether a given model has low outlier balanced accuracy
  mean_balacc <- mean(oob_results$OOB_BalancedAcc)
  sd_balacc <- sd(oob_results$OOB_BalancedAcc)
  oob_results$OOB_BalancedAccOutlier <- as.factor(ifelse(oob_results$OOB_BalancedAcc < mean_balacc - 1.5*sd_balacc,1,0))
  levels(oob_results$OOB_BalancedAccOutlier) <- c("0","1")
  
  #create ensemble of the non-low-outlier-accuracy models
  #create field of average probability
  bagged_preds$ensemble_prob <- apply(bagged_preds[,c(oob_results$Iteration[oob_results$OOB_BalancedAccOutlier==0])]
                                      ,1
                                      ,function(x) mean(x))
  #if usepriors==TRUE, use prior positive prevalence
  #else, use 0.5
  if (usepriors==TRUE) {
    bagged_preds$ensemble_y <- as.factor(ifelse(bagged_preds$ensemble_prob>=prior_positive,1,0))
  } else {
    bagged_preds$ensemble_y <- as.factor(ifelse(bagged_preds$ensemble_prob>=0.50,1,0))
  }
  levels(bagged_preds$ensemble_y) <- c("0","1")
  
  print(paste0("Ensemble created. "
               ,length(oob_results$Iteration[oob_results$OOB_BalancedAccOutlier==0])
               ," models used, "
               ,length(oob_results$Iteration[oob_results$OOB_BalancedAccOutlier==1])
               ," model(s) dropped as low accuracy outlier(s)."))
  print(paste0("Average OOB sensitivity: ",round(100*mean(oob_results$OOB_Sensitivity[oob_results$OOB_BalancedAccOutlier==0]),1),"%."))
  print(paste0("Average OOB specificity: ",round(100*mean(oob_results$OOB_Specificity[oob_results$OOB_BalancedAccOutlier==0]),1),"%."))
  print(paste0("Average OOB balanced accuracy: ",round(100*mean(oob_results$OOB_BalancedAcc[oob_results$OOB_BalancedAccOutlier==0]),1),"%."))
  print(paste0("Average false positives: ",mean(oob_results$OOB_FalsePos)
               ,". Average true positives: ",mean(oob_results$OOB_TruePos),"."))
  print(paste0("Average false negatives: ",mean(oob_results$OOB_FalseNeg)
               ,". Average true negatives: ",mean(oob_results$OOB_TrueNeg),"."))
  
  return(c(oob_results,bagged_preds,varimp))
}