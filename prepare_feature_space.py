from feature_extraction import data, extractor
import os

data = data.load_pickled_data()
outPath = os.path.join("feature_extraction", "_data_sets/test" + ".pkl")
extractor.prepare_data_pickle(outPath)
