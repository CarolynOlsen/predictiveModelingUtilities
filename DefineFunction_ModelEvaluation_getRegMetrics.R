# PURPOSE: 
# This function takes a caret train object as its input,
# scores a new data set, and returns metrics for a continuous 
# outcome.
# AUTHOR: Carolyn Olsen
# DATE: 2-6-2018

getRegMetrics <- function(model, newdata, y_name) {
  
  pred_y <- predict(model,
                    newdata=newdata,
                    na.action=na.pass)
  
  options(scipen=999) #i hate scientific notation
  
  library(MLmetrics)
  rmse <- RMSE(y_pred, newdata[[y_name]])
  r2 <- R2_Score(y_pred, newdata[[y_name]])
  r2adj <- r2 - (1-r2)*(length(model$coefnames) / (nrow(newdata) - length(model$coefnames) - 1))
  mae <- MAE(y_pred, newdata[[y_name]])
  mape <- MAPE(y_pred, newdata[[y_name]])
  rho <- cor.test(y_pred, newdata[[y_name]], method='spearman')$estimate
  
  metrics <- data.frame(
    metric = c("RMSE","R-squared","Adj R-squared","MAE","MAPE","Rank correlation Rho")
    , value = c(rmse, r2, r2adj, mae, mape, rho)
  )
  
  return(metrics)
}
