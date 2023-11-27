"""
Colton Harris
D&D Character Creation Analysis Testing File

Tests all the setup code and generates 5 plots
using 2 small subsets of character data.
Prints "Done!" to the terminal when finished.
"""
from cse163_utils import assert_equals

import pandas as pd
import geopandas as gpd
import data_setup as ds
import characters_by_nationality as cbn
import class_prediction as cp
import torch


def test_q1_setups(char_data: pd.DataFrame, num_chars: int,
                   geo_char_data: gpd.GeoDataFrame) -> None:
    """
    Asserts that the provided character data has num_chars
    many characters (rows). Assertts that the geo char data
    has one more col and the same number of rows as the char
    data. If these are true than filtering is occuring
    as expected for the data setup methods

    Args:
        char_data (pd.DataFrame): Filterd D&D character data to be verified
        num_chars (int): The number of expected characters in char_data
        geo_char_data (gpd.GeoDataFrame): Filterd D&D geo data to be verified
    """
    assert_equals(num_chars, char_data.shape[0])
    assert_equals(geo_char_data.shape[1], char_data.shape[1] + 1)
    assert_equals(geo_char_data.shape[0], char_data.shape[0])


def test_q1_cw(cw: gpd.gpd.GeoDataFrame,
               weapons_unique: int,
               countries_unique: int) -> None:
    """
    Asserts that the provided country and weapons character data has
    weapons_unique many unqiue weapons and countries_unique many
    unique countries.

    Args:
        cw (gpd.GeoDataFrame): Filterd D&D gcharacter, weapons, and countries
        weapons_unique (int): The num of expected unique weapons
        countries_unique (gpd.GeoDataFrame): The num of expected
                                             unique countries
    """
    assert_equals(weapons_unique, cw['processedWeapons'].unique().shape[0])
    assert_equals(countries_unique, cw['country'].unique().shape[0])


def test_get_tensors() -> None:
    """
    Asserts that the tensor created with the character data
    from dnd_chars_test1/tsv has the expected list of tensor
    tuples.
    """
    test_char_data = ds.init_char_data('test_data/dnd_chars_test2.tsv')
    data, class_map = ds.get_tensor_data(test_char_data)
    expected_X = [[53/450, 13/30, 8/30, 15/30, 10/30, 15/30, 16/30, 13/30],
                  [55/450, 16/30, 8/30, 16/30, 14/30, 18/30, 10/30, 8/30],
                  [24/450, 16/30, 8/30, 14/30, 14/30, 17/30, 12/30, 10/30]]
    expected_X = torch.tensor(expected_X, dtype=torch.float32)
    expected_y = [torch.tensor(1), torch.tensor(0), torch.tensor(0)]
    expected = [(X, y) for X, y in zip(expected_X, expected_y)]
    for i in range(len(expected)):
        assert_equals(True, torch.equal(expected[i][0], data[i][0]))
        assert_equals(True, torch.equal(expected[i][1], data[i][1]))


def main():
    # testing file paths
    map_file_path = 'data/ne_10m_admin_0_countries.shp'
    char_file_path = 'test_data/dnd_chars_test1.tsv'
    # Create and Filter inidividual data sets
    map_data = ds.init_map_data(map_file_path)
    char_data = ds.init_char_data(char_file_path)
    geo_char_data = ds.init_geo_char_data(char_data, map_data)

    # Verify data sets were created correctly and filtered as expected
    test_q1_setups(char_data, 16, geo_char_data)

    # Get countries and weapons (and geometry)
    cw: gpd.GeoDataFrame = ds.get_country_weapons(geo_char_data)

    # Test cw
    test_q1_cw(cw, 3, 1)

    # Test Generate National Analysis Plots
    # Should only plot Canada. Simple weapons should have one more
    # melee and ranged is identical.
    # martial = 9/16, half-casters = 4/16, full-casters=5/16
    cbn.plot_ranged_weapons(cw, map_data)
    cbn.plot_melee_weapons(cw, map_data)
    cbn.plot_class_archetypes(geo_char_data, map_data)

    # Generate Class Analysis Plots
    # Ranger = 1, Druid = 4, Fighter = 9, Paladin = 1
    cp.plot_classes(char_data)

    test_get_tensors()
    print('Done!')


if __name__ == '__main__':
    main()
