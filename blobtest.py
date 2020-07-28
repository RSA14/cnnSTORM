from skimage import io, data, feature
import pandas as pd
import matplotlib.pyplot as plt
import image_processing
import data_processing

# simulated = io.imread("ThunderSTORM/Simulated/LowDensity/0.tif")
# simulated_truth = pd.read_csv("ThunderSTORM/Simulated/LowDensity/0.csv")
# simulated_test = feature.blob_log(simulated, threshold=0.001)
# print(simulated_test)
# actual = io.imread("ThunderSTORM/purePSF_5_MMStack_Pos0.ome.tif")
# actual_test = feature.blob_log(actual, threshold = 0.005)
# print(actual_test)
#

# MT = io.imread("sequence-as-stack-MT1.N1.LD-AS-Exp.tif")
# MT_truth = pd.read_csv("positions.csv", header=0)
#
# MT_truth = pd.DataFrame({"x": round(MT_truth["x"] / 100), "y": round(MT_truth["y"] / 100),
#                          "z": MT_truth["z"]})


im = io.imread('ThunderSTORM/purePSF_5_MMStack_Pos0.ome.tif')
test = pd.read_csv('ThunderSTORM/purePSF_5_MMStack_Pos0.ome.csv')
# im1 = im[0]

# psfs = data_processing.process_blobs(im1, min_sigma=5, threshold=0.005)

# x_,y_ = data_processing.process_STORM_zstack(im, test, (-1000,1000,40), intensity_threshold= 1000000,
#                                              bound=16)

x_, y_ = data_processing.process_blob_zstack(im, z_data=(-1000, 1000, 40), threshold=0.005)

print(x_.shape, y_.shape)
io.imshow(x_[0].reshape(32, 32))
plt.show()
