from data_processing import generate_STORM_data

X_train, y_train = generate_STORM_data('ThunderSTORM/Simulated/LowDensity',
                                       samples=32)