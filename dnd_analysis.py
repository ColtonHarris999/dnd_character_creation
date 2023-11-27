"""
Colton Harris
D&D Character Creation Analysis

Main runnable program for this project.
Will save 7 files to the folder this file is in.
Takes about a minute to run and will print "Done!"
when finished.
Starts by plotting national relationships, then trains
the netowrk, then plots the accuracy of the network.
"""


import geopandas as gpd
import data_setup as ds
import characters_by_nationality as cbn
import class_prediction as cp
from class_prediction import Net
from sklearn.model_selection import train_test_split


def main():
    # file paths
    map_file_path = 'data/ne_10m_admin_0_countries.shp'
    char_file_path = 'data/dnd_chars_all.tsv'
    # Create and Filter inidividual data sets
    map_data = ds.init_map_data(map_file_path)
    char_data = ds.init_char_data(char_file_path)
    geo_char_data = ds.init_geo_char_data(char_data, map_data)

    # Get countries and weapons (and geometry)
    cw: gpd.GeoDataFrame = ds.get_country_weapons(geo_char_data)

    # Generate National Analysis Plots
    cbn.plot_ranged_weapons(cw, map_data)
    cbn.plot_melee_weapons(cw, map_data)
    cbn.plot_class_archetypes(geo_char_data, map_data)

    # Generate Class Analysis Plots
    cp.plot_classes(char_data)

    data, class_map = ds.get_tensor_data(char_data)
    train_data, test_data = train_test_split(data, test_size=0.2)
    net: Net = cp.train(train_data, test_data)

    # plot accuracy & inaccuracy
    cp.plot_class_accuracy(net, data, class_map)
    print('Done!')


if __name__ == '__main__':
    main()
