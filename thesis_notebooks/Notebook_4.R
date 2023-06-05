mon_rean = read.csv('C:\\Users\\conor\\General_College\\UTRECHT ADS\\Thesis\\Data\\Ameriflux_docs\\ameri_monthly_reanalysis_by_hour_mean0.csv')
mon_rean <- mon_rean[-which(mon_rean$site == 'CA-LP1'),]

#CV on site data w/ mean = 0
df_cv_1 <- data.frame(site = 'placeholder', prediction = 0, year.month = '1900-00', hour = 999)
#situ = 'US-Tw1'

for (situ in unique(mon_rean$site)){
  print(situ)
  
  unfold_i = mon_rean[which(mon_rean$site != situ),]
  fold_i = mon_rean[which(mon_rean$site == situ),]
  
  RF_cv <- randomForest::randomForest(NEE_VUT_REF_subt ~ TA_ERA + SW_IN_ERA + LW_IN_ERA + PA_ERA + P_ERA + LE_F_MDS + H_F_MDS + RH, 
                                      data = unfold_i, ntree = 100)
  
  fold_preds <-  predict(RF_cv, fold_i[ c('TA_ERA','SW_IN_ERA','LW_IN_ERA','PA_ERA','P_ERA','LE_F_MDS','H_F_MDS','RH') ]  )
  cv_vec <- data.frame(site =  rep(situ,length(fold_preds)), prediction =  fold_preds, 
                       year.month = fold_i$year.month, hour =  fold_i$hour )
  df_cv_1 <- rbind(df_cv_1, cv_vec)
  
} 

df_cv_1 <- df_cv_1[-1,]

merged_juan <- dplyr::inner_join(mon_rean, df_cv_1, 
                                 by = c("site" = "site", 
                                        "year.month" = "year.month",
                                        'hour' = 'hour'))

write.csv(merged_juan, 'C:\\Users\\conor\\General_College\\UTRECHT ADS\\Thesis\\Data\\Ameriflux\\mean_0_RF_preds_1.csv') 