#Here, the 00 will need to be changed to all hours 00-23

#predictions for june 2018 00am:
june18_00am = read.csv('C:\\Users\\conor\\General_College\\UTRECHT ADS\\Thesis\\Data\\rescale_dfs\\june_2018\\june_2018_00am_ERA.csv')


june18_00am$flux_preds <-  predict(amri_rf, june18_00am)
write.csv(june18_00am, 'C:\\Users\\conor\\General_College\\UTRECHT ADS\\Thesis\\Data\\rescale_dfs\\june_2018_00am_ERA_preds.csv')
