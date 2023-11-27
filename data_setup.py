"""
Colton Harris
D&D Character Creation Data Setup

Initializes the data used for the rest of the project:
1. Character data
2. Map data
3. Character & Map data
4. Character's weapons and player's country
5. Tensor data for training the network
"""

import pandas as pd
import geopandas as gpd
import torch
from sklearn.preprocessing import LabelEncoder


def init_char_data(char_file_path: str) -> pd.DataFrame:
    """
    Filters the data at the given path to create a DataFrame
    representing the different D&D characters players have made.
    Filters out all characters with any of the following:
     - Level less than 1 or greater than 20
     - Any stat less than 1 or greater than 30
     - AC less than 1

    Args:
        char_file_path (str): File path to tsv character data

    Returns:
        pd.DataFrame: filtered Datafram representing D&D characters
    """
    # Get and filter char data
    char_data = pd.read_csv(char_file_path, delimiter='\t',
                            on_bad_lines='skip')
    # remove unecessary rows and drop na
    char_data = char_data.loc[:, 'race':'processedWeapons']
    # remove levels < 1 and > 20
    min_level_mask = char_data['level'] >= 1
    max_level_mask = char_data['level'] <= 20
    char_data = char_data[min_level_mask & max_level_mask]
    # remove stats < 1 and > 30
    stats = ['Str', 'Dex', 'Con', 'Int', 'Wis', 'Cha']
    for stat in stats:
        min_stats_mask = char_data[stat] >= 1
        max_stats_mask = char_data[stat] <= 30
        char_data = char_data[min_stats_mask & max_stats_mask]
    # remove AC < 1
    char_data = char_data[char_data['AC'] >= 1]
    return char_data


def init_geo_char_data(char_data: pd.DataFrame,
                       map_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Merges the country geometry data into the character data.

    Args:
        char_data (pd.DataFrame): filtered character data
        map_data (gdp.GeoDataFrame): geospatial map data to be added in

    Returns:
        gpd.GeoDataFrame: filtered GeoDatafram representing D&D characters
    """
    # drop na countries
    # char_data = char_data[~char_data['country'].isna()]
    char_data = char_data.dropna(subset=['country'])
    # Add in geospatial data from map_data
    return map_data.merge(char_data, on='country', how='inner')


def init_map_data(map_file_path: str) -> gpd.GeoDataFrame:
    """
    Creates and filters a GeoDataFrame with the data at the given path.
    Data is filtered to only include country and geometry information.
    Changes some country names to match those used in the character data.
    Renames 'NAME_EN' column to 'country'

    Args:
        map_file_path (str): File path to geospatial data

    Returns:
        gpd.GeoDataFrame: Filtered map data
    """
    map_data = gpd.read_file(map_file_path)
    # filter down
    map_data = map_data[['NAME_EN', 'geometry']].dropna()
    # Fix mismatch names
    map_data['NAME_EN'] = map_data['NAME_EN'].str.replace(
        'United States of America', 'United States')
    # rename NAME_EN to country
    map_data.rename(columns={'NAME_EN': 'country'}, inplace=True)
    map_data = map_data[map_data['country'] != 'Antarctica']
    return map_data


def get_country_weapons(char_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Given a GeoDataFrame of character data, filters down to and returns
    just the country of the creator, character weapons, and geometry.
    Only countries with at least 15 entries will be included.

    Args:
        char_data (gpd.GeoDataFrame): D&D characters and corresponding map data

    Returns:
        gpd.GeoDataFrame: Filtered character data
    """
    # Filter to just countries and weapons (and geometry)
    cw = char_data[['country', 'processedWeapons', 'geometry']]
    cw = gpd.GeoDataFrame(cw.dropna())

    # filter out countries with less than 15 entries:
    country_counts = cw.groupby('country')['country'].count()
    country_counts = country_counts[country_counts >= 15]
    return cw[cw['country'].isin(country_counts.index)]


def get_tensor_data(char_data: pd.DataFrame) -> (
                    torch.Tensor, torch.Tensor, dict[int, str]):
    """
    Converts the given D&D character dataset into training/testing
    data for the class predictor nueral network. Creates tensors
    for the features (HP, AC, Str, Dex, Con, Int, Wis, Cha) and the
    labels (the class). Also creates a dictionary to convert encoded
    labels back into class name.

    Args:
        char_data (pd.DataFrame): D&D character dataset

    Returns:
        (torch.Tensor, torch.Tensor, dict[int, str]): Tuple consisting of
        the features tensor, labels tensor, and the dictionary to convert
        the encoded labels back into classes.
    """
    # Classes
    martial = {'Barbarian', 'Fighter', 'Rogue', 'Monk'}
    half_casters = {'Artificer', 'Ranger', 'Paladin'}
    full_casters = {'Bard', 'Cleric', 'Druid', 'Sorcerer', 'Warlock', 'Wizard'}
    all_classes = martial | half_casters | full_casters

    # Attributes used in machine learning model
    attributes = ['HP', 'AC', 'Str', 'Dex', 'Con', 'Int', 'Wis', 'Cha']

    # Filter to just attributes and class, remove multiclasses and homebrew
    data = (char_data[(attributes + ['justClass'])]).dropna()
    data = data[data['justClass'].isin(all_classes)]

    # Convert values to 0-1
    data['HP'] = data['HP'] / 450  # Max HP
    attributes.remove('HP')

    for attribute in attributes:
        data[attribute] = data[attribute] / 30

    # labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(data['justClass'])
    # labels = encoded_labels.tolist()
    labels = torch.tensor(encoded_labels, dtype=torch.int64)

    # labels = torch.tensor(encoded_labels, dtype=torch.int64)

    # Create map of encoded labels back to class names
    label_mapping = {index: class_name for index, class_name in enumerate(
        label_encoder.classes_)}

    # features
    features = data.drop(columns=['justClass'])
    features = torch.tensor(features.values, dtype=torch.float32)

    data_tuples = list(zip(features, labels))

    # return features, labels, label_mapping
    return data_tuples, label_mapping
