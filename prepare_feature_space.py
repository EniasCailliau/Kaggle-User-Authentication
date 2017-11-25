from feature_extraction import data, extractor
import os

# data.create_pickled_data()
data = data.load_pickled_data()
outPath = os.path.join("feature_extraction", "_data_sets/test2" + ".pkl")
extractor.prepare_data_pickle(outPath)
