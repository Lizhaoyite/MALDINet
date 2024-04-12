# install.packages("MALDIquant", repos = "http://mirror.lzu.edu.cn/CRAN/")
# install.packages("MALDIquantForeign", repos = "http://mirror.lzu.edu.cn/CRAN/")
suppressWarnings(library(MALDIquant))
suppressWarnings(library(MALDIquantForeign))

#############Define Functions###################

#Data Process
Process <- function(s,minFrequency,tolerance = 0.002){
  #Variance Stabilization
  s <- transformIntensity(s,method="sqrt")
  
  # Smoothing
  s <- smoothIntensity(s,method="SavitzkyGolay",
                       halfWindowSize=10)
  
  #Baseline Correction
  s <- removeBaseline(s,method="TopHat")
  
  #Intensity Calibration/Normalization
  s <- calibrateIntensity(s,method="TIC")
  
  #SNR
  s <- detectPeaks(s, method="MAD",
                   halfWindowSize=10, SNR=2)
  
  #Peak Binning
  s <- binPeaks(s, tolerance=tolerance)
  
  #Fliter Nan
  s <- filterPeaks(s, minFrequency=minFrequency)
  s
}

args <- commandArgs(trailingOnly = TRUE)

#####修改工作路径为当前路径####
# setwd("C:/Users/Jiao/Downloads/Compressed/R-MDL")

current_path <- getwd()
s_input <- importMzMl(args[1],verbose=FALSE)
print("File is already read successfully.")

####################Data Process##################
fre <- as.numeric(ifelse(length(args) >= 2, args[2], 0.2))
print(paste("The value of minFreq is:", fre))
peaks_all <- Process(s_input,minFrequency = fre)
print("File is already processed successfully.")
#####################Export Matrix##################
featureMatrix <- intensityMatrix(peaks_all)
write.csv(featureMatrix,file = file.path(current_path, "featureMatrix.csv"),
          row.names = FALSE)

print("File is already written successfully.")

