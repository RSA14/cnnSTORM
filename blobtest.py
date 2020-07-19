from skimage import io, data, feature
import pandas as pd
import matplotlib.pyplot as plt
import image_processing

simulated = io.imread("ThunderSTORM/Simulated/LowDensity/0.tif")
simulated_truth = pd.read_csv("ThunderSTORM/Simulated/LowDensity/0.csv")
simulated_test = feature.blob_log(simulated, threshold=0.001)
print(simulated_test)
actual = io.imread("ThunderSTORM/purePSF_5_MMStack_Pos0.ome.tif")
actual_test = feature.blob_log(actual, threshold = 0.005)
print(actual_test)
