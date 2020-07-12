from data_processing import process_STORM_data

X_train, y_train = process_STORM_data('ThunderSTORM/Simulated/LowDensity',
                                      samples=32)