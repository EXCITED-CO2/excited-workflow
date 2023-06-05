set.seed(1)
amri_rf1 <- randomForest::randomForest(NEE_VUT_REF_subt ~ TA_ERA + SW_IN_ERA + LW_IN_ERA + PA_ERA + P_ERA + LE_F_MDS + H_F_MDS + RH, 
                                       data = mon_rean, ntree = 100)



df_00_jan <- read.csv('df_00_jan.csv')
df_00_jan <- df_00_jan %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


jan_00_preds <- predict(amri_rf1, df_00_jan[ , -which(names(df_00_jan) %in% c('year', 'month')) ]  )
jan_00_preds_mean <- mean(jan_00_preds)*10^(-6)
#
df_00_feb <- read.csv('df_00_feb.csv')
df_00_feb <- df_00_feb %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


feb_00_preds <- predict(amri_rf1, df_00_feb[ , -which(names(df_00_feb) %in% c('year', 'month')) ]  )
feb_00_preds_mean <- mean(feb_00_preds)*10^(-6)
#
df_00_mar <- read.csv('df_00_mar.csv')
df_00_mar <- df_00_mar %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


mar_00_preds <- predict(amri_rf1, df_00_mar[ , -which(names(df_00_mar) %in% c('year', 'month')) ]  )
mar_00_preds_mean <- mean(mar_00_preds)*10^(-6)
#
df_00_apr <- read.csv('df_00_apr.csv')
df_00_apr <- df_00_apr %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


apr_00_preds <- predict(amri_rf1, df_00_apr[ , -which(names(df_00_apr) %in% c('year', 'month')) ]  )
apr_00_preds_mean <- mean(apr_00_preds)*10^(-6)
#
df_00_may <- read.csv('df_00_may.csv')
df_00_may <- df_00_may %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


may_00_preds <- predict(amri_rf1, df_00_may[ , -which(names(df_00_may) %in% c('year', 'month')) ]  )
may_00_preds_mean <- mean(may_00_preds)*10^(-6)
# 
df_00_jun <- read.csv('df_00_jun.csv')
df_00_jun <- df_00_jun %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


jun_00_preds <- predict(amri_rf1, df_00_jun[ , -which(names(df_00_jun) %in% c('year', 'month')) ]  )
jun_00_preds_mean <- mean(jun_00_preds)*10^(-6)
#
df_00_jul <- read.csv('df_00_jul.csv')
df_00_jul <- df_00_jul %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


jul_00_preds <- predict(amri_rf1, df_00_jul[ , -which(names(df_00_jul) %in% c('year', 'month')) ]  )
jul_00_preds_mean <- mean(jul_00_preds)*10^(-6)
#
df_00_aug <- read.csv('df_00_aug.csv')
df_00_aug <- df_00_aug %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


aug_00_preds <- predict(amri_rf1, df_00_aug[ , -which(names(df_00_aug) %in% c('year', 'month')) ]  )
aug_00_preds_mean <- mean(aug_00_preds)*10^(-6)
#
df_00_sep <- read.csv('df_00_sep.csv')
df_00_sep <- df_00_sep %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


sep_00_preds <- predict(amri_rf1, df_00_sep[ , -which(names(df_00_sep) %in% c('year', 'month')) ]  )
sep_00_preds_mean <- mean(sep_00_preds)*10^(-6)
#
df_00_oct <- read.csv('df_00_oct.csv')
df_00_oct <- df_00_oct %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


oct_00_preds <- predict(amri_rf1, df_00_oct[ , -which(names(df_00_oct) %in% c('year', 'month')) ]  )
oct_00_preds_mean <- mean(oct_00_preds)*10^(-6)
#
df_00_nov <- read.csv('df_00_nov.csv')
df_00_nov <- df_00_nov %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


nov_00_preds <- predict(amri_rf1, df_00_nov[ , -which(names(df_00_nov) %in% c('year', 'month')) ]  )
nov_00_preds_mean <- mean(nov_00_preds)*10^(-6)
#
df_00_dec <- read.csv('df_00_dec.csv')
df_00_dec <- df_00_dec %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


dec_00_preds <- predict(amri_rf1, df_00_dec[ , -which(names(df_00_dec) %in% c('year', 'month')) ]  )
dec_00_preds_mean <- mean(dec_00_preds)*10^(-6)

flux_preds_00 <- data.frame(flux_predictions = c(jan_00_preds_mean,feb_00_preds_mean,mar_00_preds_mean,apr_00_preds_mean,may_00_preds_mean,jun_00_preds_mean,
                                                 jul_00_preds_mean,aug_00_preds_mean,sep_00_preds_mean,oct_00_preds_mean,nov_00_preds_mean,dec_00_preds_mean),
                            month = c('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'),
                            year = rep(2000,12))
write.csv(flux_preds_00, 'jun_s_00_1.csv', row.names = F)



##################################################################################################################################
                            #################################################################
##################################################################################################################################

set.seed(1)
amri_rf_non0 <- randomForest::randomForest(NEE_VUT_REF_x ~ TA_ERA + SW_IN_ERA + LW_IN_ERA + PA_ERA + P_ERA + LE_F_MDS + H_F_MDS + RH, 
                                           data = mon_rean, ntree = 100)


df_00_jan <- read.csv('df_00_jan.csv')
df_00_jan <- df_00_jan %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


jan_00_preds <- predict(amri_rf_non0, df_00_jan[ , -which(names(df_00_jan) %in% c('year', 'month')) ]  )
jan_00_preds_mean <- mean(jan_00_preds)*10^(-6)
#
df_00_feb <- read.csv('df_00_feb.csv')
df_00_feb <- df_00_feb %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


feb_00_preds <- predict(amri_rf_non0, df_00_feb[ , -which(names(df_00_feb) %in% c('year', 'month')) ]  )
feb_00_preds_mean <- mean(feb_00_preds)*10^(-6)
#
df_00_mar <- read.csv('df_00_mar.csv')
df_00_mar <- df_00_mar %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


mar_00_preds <- predict(amri_rf_non0, df_00_mar[ , -which(names(df_00_mar) %in% c('year', 'month')) ]  )
mar_00_preds_mean <- mean(mar_00_preds)*10^(-6)
#
df_00_apr <- read.csv('df_00_apr.csv')
df_00_apr <- df_00_apr %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


apr_00_preds <- predict(amri_rf_non0, df_00_apr[ , -which(names(df_00_apr) %in% c('year', 'month')) ]  )
apr_00_preds_mean <- mean(apr_00_preds)*10^(-6)
#
df_00_may <- read.csv('df_00_may.csv')
df_00_may <- df_00_may %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


may_00_preds <- predict(amri_rf_non0, df_00_may[ , -which(names(df_00_may) %in% c('year', 'month')) ]  )
may_00_preds_mean <- mean(may_00_preds)*10^(-6)
# 
df_00_jun <- read.csv('df_00_jun.csv')
df_00_jun <- df_00_jun %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


jun_00_preds <- predict(amri_rf_non0, df_00_jun[ , -which(names(df_00_jun) %in% c('year', 'month')) ]  )
jun_00_preds_mean <- mean(jun_00_preds)*10^(-6)
#
df_00_jul <- read.csv('df_00_jul.csv')
df_00_jul <- df_00_jul %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


jul_00_preds <- predict(amri_rf_non0, df_00_jul[ , -which(names(df_00_jul) %in% c('year', 'month')) ]  )
jul_00_preds_mean <- mean(jul_00_preds)*10^(-6)
#
df_00_aug <- read.csv('df_00_aug.csv')
df_00_aug <- df_00_aug %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


aug_00_preds <- predict(amri_rf_non0, df_00_aug[ , -which(names(df_00_aug) %in% c('year', 'month')) ]  )
aug_00_preds_mean <- mean(aug_00_preds)*10^(-6)
#
df_00_sep <- read.csv('df_00_sep.csv')
df_00_sep <- df_00_sep %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


sep_00_preds <- predict(amri_rf_non0, df_00_sep[ , -which(names(df_00_sep) %in% c('year', 'month')) ]  )
sep_00_preds_mean <- mean(sep_00_preds)*10^(-6)
#
df_00_oct <- read.csv('df_00_oct.csv')
df_00_oct <- df_00_oct %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


oct_00_preds <- predict(amri_rf_non0, df_00_oct[ , -which(names(df_00_oct) %in% c('year', 'month')) ]  )
oct_00_preds_mean <- mean(oct_00_preds)*10^(-6)
#
df_00_nov <- read.csv('df_00_nov.csv')
df_00_nov <- df_00_nov %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


nov_00_preds <- predict(amri_rf_non0, df_00_nov[ , -which(names(df_00_nov) %in% c('year', 'month')) ]  )
nov_00_preds_mean <- mean(nov_00_preds)*10^(-6)
#
df_00_dec <- read.csv('df_00_dec.csv')
df_00_dec <- df_00_dec %>%
  rename("RH" = "X2.metre.dewpoint.temperature" ,
         "TA_ERA" = "X2.metre.temperature" ,
         "LE_F_MDS" = "Surface.latent.heat.flux",
         "SW_IN_ERA" ="Surface.net.solar.radiation" ,
         "LW_IN_ERA" = "Surface.net.thermal.radiation" ,
         "H_F_MDS" = "Surface.sensible.heat.flux" ,
         "PA_ERA" = "Surface.pressure" ,
         "P_ERA" ="Total.precipitation" )


dec_00_preds <- predict(amri_rf_non0, df_00_dec[ , -which(names(df_00_dec) %in% c('year', 'month')) ]  )
dec_00_preds_mean <- mean(dec_00_preds)*10^(-6)

flux_preds_00 <- data.frame(flux_predictions = c(jan_00_preds_mean,feb_00_preds_mean,mar_00_preds_mean,apr_00_preds_mean,may_00_preds_mean,jun_00_preds_mean,
                                                 jul_00_preds_mean,aug_00_preds_mean,sep_00_preds_mean,oct_00_preds_mean,nov_00_preds_mean,dec_00_preds_mean),
                            month = c('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'),
                            year = rep(2000,12))
write.csv(flux_preds_00, 'non0_jun_s_00.csv', row.names = F)

