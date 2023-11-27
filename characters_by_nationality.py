"""
Colton Harris
CSE 163
D&D Character Creation by Nationality

Plots the national relationships for character data:
1. % Each country uses ranged weapons
2. % Each country uses melee weapons
2. % Of eaach country who play specific class archetypes
   (martial, half-caster, full-caster)
"""


import geopandas as gpd
import matplotlib.pyplot as plt


def _contains_within(char_options: str, target_options: list[str]) -> bool:
    """
    Given a string of options possessed by the character and a list of the
    all the target options, returns true if the char possesses at least
    one option within the target options list.

    Args:
        char_options (str): a string with options seperated by |
        target_options (list[str]): the list of target options

    Returns:
        bool: _description_
    """
    char_options = char_options.split("|")
    for option in char_options:
        if option in target_options:
            return True
    return False


def plot_ranged_weapons(cw: gpd.GeoDataFrame,
                        map_data: gpd.GeoDataFrame) -> None:
    """
    Plots a world heat map where each country's heat is the percentage
    of characters created by players in that country who use at least
    one ranged weapon from the Players Handbook. A country must have at
    least 15 entries to be considered. The generated figure is two graphs,
    one of all ranged weapons, and one of just martial ranged weapons.
    Saves the file as 'percent_ranged.png'

    Args:
        cw (gpd.GeoDataFrame): dataset of just creator's Country, character's
                               Weapons, and geometry of the country.
        map_data (gpd.GeoDataFrame): map data for every country
    """
    all_ranged_weapons = {'Crossbow, Light', 'Longbow', 'Shortbow', 'Javelin',
                          'Dart', 'Sling', 'Blowgun', 'Crossbow, Hand',
                          'Crossbow, Heavy', 'Net'}
    martial_ranged_weapons = {'Longbow', 'Blowgun', 'Crossbow, Hand',
                              'Crossbow, Heavy', 'Net'}

    # Check ranged weapons and drop processedWeapons
    cw['ranged_all'] = cw.apply(
        lambda row: _contains_within(
            row['processedWeapons'], all_ranged_weapons), axis=1)
    cw['ranged_martial'] = cw.apply(
        lambda row: _contains_within(
            row['processedWeapons'], martial_ranged_weapons), axis=1)
    cw = gpd.GeoDataFrame(cw[['country', 'ranged_all',
                              'ranged_martial', 'geometry']])

    # groupby country and find percentage using ranged weapons
    # use groupby becuase its MUCH faster than dissolve in this case
    cw = gpd.GeoDataFrame(cw.groupby('country').agg(
        {'ranged_all': 'mean', 'ranged_martial': 'mean',
         'geometry': 'first'}))
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # All ranged
    map_data.plot(ax=ax1, color='#DDDDDD', edgecolor='#FFFFFF')
    cw.plot(ax=ax1, column='ranged_all', legend=True, cmap='coolwarm')
    plt.title('% D&D Characters Using Any Ranged Weapon')
    ax1.set_title('% D&D Characters Using\nRanged Weapons')
    # Martial ranged
    map_data.plot(ax=ax2, color='#DDDDDD', edgecolor='#FFFFFF')
    cw.plot(ax=ax2, column='ranged_martial', legend=True, cmap='coolwarm')
    ax2.set_title('% D&D Characters Using\nMartial Ranged Weapons')
    # save
    fig.suptitle('Percent Ranged Weapons by Country')
    plt.savefig('percent_ranged.png', bbox_inches="tight")


def plot_melee_weapons(cw: gpd.GeoDataFrame,
                       map_data: gpd.GeoDataFrame) -> None:
    """
    Plots a world heat map where each country's heat is the percentage
    of characters created by players in that country who use at least
    one melee weapon from the Players Handbook. A country must have at
    least 15 entries to be considered. The generated figure is two graphs,
    one of all melee weapons, and one of just martial ranged weapons.
    Saves the file as 'percent_ranged.png'

    Args:
        cw (gpd.GeoDataFrame): dataset of just creator's Country, character's
                               Weapons, and geometry of the country.
        map_data (gpd.GeoDataFrame): map data for every country
    """
    simple_melee_weapons = {'Club', 'Dagger', 'Greatclub', 'Handaxe',
                            'Sickle', 'Spear'}
    martial_melee_weapons = {'Battleaxe', 'Flail', 'Glaive', 'Greataxe',
                             'Greatsword', 'Halberd', 'Lance', 'Longsword',
                             'Maul', 'Morningstar', 'Pike', 'Rapier',
                             'Scimitar', 'Shortsword', 'Trident', 'War pick',
                             'Warhammer', 'Whip'}
    all_melee_weapons = simple_melee_weapons | martial_melee_weapons

    # Check melee weapons and drop processedWeapons
    cw['melee_all'] = cw.apply(
        lambda row: _contains_within(
            row['processedWeapons'], all_melee_weapons), axis=1)
    cw['melee_martial'] = cw.apply(
        lambda row: _contains_within(
            row['processedWeapons'], martial_melee_weapons), axis=1)
    cw = gpd.GeoDataFrame(cw[['country', 'melee_all',
                              'melee_martial', 'geometry']])

    # groupby country and find percentage using melee weapons
    # use groupby becuase its MUCH faster than dissolve in this case
    cw = gpd.GeoDataFrame(cw.groupby('country').agg(
        {'melee_all': 'mean', 'melee_martial': 'mean',
         'geometry': 'first'}))
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # All ranged
    map_data.plot(ax=ax1, color='#DDDDDD', edgecolor='#FFFFFF')
    cw.plot(ax=ax1, column='melee_all', legend=True, cmap='coolwarm')
    plt.title('% D&D Characters Using Any Melee Weapon')
    ax1.set_title('% D&D Characters Using\nMelee Weapons')
    # Martial ranged
    map_data.plot(ax=ax2, color='#DDDDDD', edgecolor='#FFFFFF')
    cw.plot(ax=ax2, column='melee_martial', legend=True, cmap='coolwarm')
    ax2.set_title('% D&D Characters Using\nMartial Melee Weapons')
    # save
    fig.suptitle('Percent Melee Weapons by Country')
    plt.savefig('percent_melee.png', bbox_inches="tight")


def plot_class_archetypes(char_data: gpd.GeoDataFrame,
                          map_data: gpd.GeoDataFrame) -> None:
    """
    Plots 3 world heat maps where each country's heat is the percentage
    of characters created by players in that country who have at least
    one level in a specific class archetype. The 3 archetypes plotted are
    martial fighters, half casters, and full casters. The country must have at
    least 15 entries to be considered.
    Saves the file as 'class_archetypes.png'

    Args:
        char_data (gpd.GeoDataFrame): D&D character dataset
        map_data (gpd.GeoDataFrame): Map data for every country
    """
    martial = {'Barbarian', 'Blood Hunter', 'Fighter', 'Rogue', 'Monk'}
    half_casters = {'Artificer', 'Ranger', 'Paladin'}
    full_casters = {'Bard', 'Cleric', 'Druid', 'Sorcerer', 'Warlock', 'Wizard'}
    # Remove uneeded data
    char_data = char_data[['justClass', 'country', 'geometry']].dropna()

    # Filter to countries with < 15 entries
    country_counts = char_data.groupby('country')['country'].count()
    country_counts = country_counts[country_counts >= 15]
    char_data = char_data[char_data['country'].isin(country_counts.index)]

    # Check melee weapons and drop processedWeapons
    char_data['martial'] = char_data.apply(
        lambda row: _contains_within(row['justClass'], martial), axis=1)
    char_data['full_caster'] = char_data.apply(
        lambda row: _contains_within(row['justClass'], full_casters), axis=1)
    char_data['half_caster'] = char_data.apply(
        lambda row: _contains_within(row['justClass'], half_casters), axis=1)
    archetypes = gpd.GeoDataFrame(char_data[['country', 'half_caster',
                                             'martial', 'full_caster',
                                             'geometry']])

    # groupby caster and find percentage
    # use groupby becuase its MUCH faster than dissolve in this case
    archetypes = gpd.GeoDataFrame(archetypes.groupby('country').agg(
        {'martial': 'mean', 'half_caster': 'mean', 'full_caster': 'mean',
         'geometry': 'first'}))
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    # Martial
    map_data.plot(ax=ax1, color='#DDDDDD', edgecolor='#FFFFFF')
    archetypes.plot(ax=ax1, column='martial', legend=True, cmap='coolwarm')
    ax1.set_title('% D&D Martial Archetype')
    # Half Caster
    map_data.plot(ax=ax2, color='#DDDDDD', edgecolor='#FFFFFF')
    archetypes.plot(ax=ax2, column='half_caster', legend=True, cmap='coolwarm')
    ax2.set_title('% D&D Half Casters')
    # Full Caster
    map_data.plot(ax=ax3, color='#DDDDDD', edgecolor='#FFFFFF')
    archetypes.plot(ax=ax3, column='full_caster', legend=True, cmap='coolwarm')
    ax3.set_title('% D&D Full Casters')
    # save
    fig.suptitle('Percent Class Archetypes by Country')
    plt.savefig('class_archetypes.png', bbox_inches="tight")
