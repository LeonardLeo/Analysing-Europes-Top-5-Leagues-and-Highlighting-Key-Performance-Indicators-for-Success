# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:01:58 2024

@author: lEO
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Creating relevant functions
def eda(dataset: pd.DataFrame, graphs: bool = False) -> dict:
    """
    Perform exploratory data analysis on the dataset.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset to perform EDA.
    graphs : bool, optional
        Choose to display exploratory data analysis visuals. The default is False.

    Returns
    -------
    dict
        A dictionary containing different evaluation metrics for exploring the 
        columns and understanding how values in the dataset are distributed.

    """
    data_unique = {}
    data_category_count = {}
    dataset.info()
    data_head = dataset.head()
    data_tail = dataset.tail()
    data_mode = dataset.mode().iloc[0]
    data_descriptive_stats = dataset.describe()
    data_more_descriptive_stats = dataset.describe(include = "all", 
                                                   datetime_is_numeric=True)
    data_correlation_matrix = dataset.corr(numeric_only = True)
    data_distinct_count = dataset.nunique()
    data_count_duplicates = dataset.duplicated().sum()
    data_count_null = dataset.isnull().sum()
    data_total_null = dataset.isnull().sum().sum()
    for each_column in dataset.columns: # Loop through each column and get the unique values
        data_unique[each_column] = dataset[each_column].unique()
    for each_column in dataset.select_dtypes(object).columns: 
        # Loop through the categorical columns and count how many values are in each category
        data_category_count[each_column] = dataset[each_column].value_counts()
        
    if graphs == True:
        # Visuals
        dataset.hist(figsize = (25, 20), bins = 10)
        plt.figure(figsize = (15, 10))
        sns.heatmap(data_correlation_matrix, annot = True, cmap = 'coolwarm')
        plt.show()
        plt.figure(figsize = (50, 30))
        sns.pairplot(dataset) # Graph of correlation across each numerical feature
        plt.show()
    
    result = {"data_head": data_head,
              "data_tail": data_tail,
              "data_mode": data_mode,
              "data_descriptive_stats": data_descriptive_stats,
              "data_more_descriptive_stats": data_more_descriptive_stats,
              "data_correlation_matrix": data_correlation_matrix,
              "data_distinct_count": data_distinct_count,
              "data_count_duplicates": data_count_duplicates,
              "data_count_null": data_count_null,
              "data_total_null": data_total_null,
              "data_unique": data_unique,
              "data_category_count": data_category_count,
              }
    return result

# Get dataset
dataset = pd.read_csv("datasets/Big 5 European football leagues teams stats.csv")

# Exploratory Data Analysis (EDA)
data_eda = eda(dataset, graphs = False)

# Descriptive Statistics
desc_stats = data_eda["data_descriptive_stats"].T.reset_index()
desc_stats["range"] = desc_stats["max"] - desc_stats["min"]
desc_stats = round(desc_stats, 2)
desc_stats.to_csv("datasets/exploratory_data_analysis_tables/Descriptive_Statistics_Table.csv", index = True)

# Missing Values
def get_percentage(row):
    total = dataset.shape[0]
    return round((row/total) * 100, 2)
    
    
missing_val = data_eda["data_count_null"]
missing_val = missing_val[missing_val > 0].reset_index()
missing_val.rename(columns = {"index": "Column", 0: "Number_Missing"}, inplace = True)
missing_val["Percentage"] = missing_val["Number_Missing"].apply(get_percentage)
missing_val.to_csv("datasets/exploratory_data_analysis_tables/Missing_Values_Table.csv", index = True)

# ---> Yellow Card - Red Card
cards = dataset[dataset["cards_red"].isna()]


# Data Cleaning and Transformation
"""
- The rank column is specified as numeric. Should be categorical
- Paris Saint German is specified as Paris S-G. Should be Paris SG
- The notes column has alot of incorrect data. The teams that qualify for the UEFA Champions
League, Europa League, and are Relegated, are not well specified across the 11 years in different
seasons. We fix by fetching accurate data online. The new source gotten to fix this issue is a 
football website called FlashFootball. Also, to draw insights from the notes column, it is split
into 3 new columns. The three new columns are:
    - UEFA Champions League
    - UEFA Europa League
    - Relegation
Their values are all Yes/No allowing us gain insight in a more organized way.
- Germany has 18 teams in the top flight while the remaining leagues have 20 teams. Also, the 
French league during the 2019-2020 season played a maximum of 28 matches. This imbalance will 
hinder certain overall analysis as this creates bias. For these, we create standardized entries
for wins, draws, losses, points, and points_per_match.
"""

# Creating standard metrics for overall unbiased analysis of Germany and France
def standardizing_wins(row):
    wins = row["wins"]
    games = row["games"]
    ratio = wins/games
    return round(ratio * 38)

def standardizing_draws(row):
    draws = row["draws"]
    games = row["games"]
    ratio = draws/games
    return round(ratio * 38)

def standardizing_losses(row):
    losses = row["losses"]
    games = row["games"]
    ratio = losses/games
    return round(ratio * 38)

def standardizing_goals_for(row):
    goals_for = row["goals_for"]
    games = row["games"]
    ratio = goals_for/games
    return round(ratio * 38)

def standardizing_goals_against(row):
    goals_against = row["goals_against"]
    games = row["games"]
    ratio = goals_against/games
    return round(ratio * 38)

def standardizing_goal_diff(row):
    goals_for = row["goals_for"]
    goals_against = row["goals_against"]
    games = row["games"]
    ratio = (goals_for/games) * 38
    ratio1 = (goals_against/games) * 38
    return round((ratio - ratio1), 2)

def standardizing_assists(row):
    assists = row["assists"]
    games = row["games"]
    ratio = assists/games
    return round(ratio * 38)

def standardizing_pens_made(row):
    pens_made = row["pens_made"]
    games = row["games"]
    ratio = pens_made/games
    return round(ratio * 38)

def standardizing_pens_att(row):
    pens_att = row["pens_att"]
    games = row["games"]
    ratio = pens_att/games
    return round(ratio * 38)

def standardizing_shots_on_target_against(row):
    shots_on_target_against = row["shots_on_target_against"]
    games = row["games"]
    ratio = shots_on_target_against/games
    return round(ratio * 38)

def standardizing_saves(row):
    saves = row["saves"]
    games = row["games"]
    ratio = saves/games
    return round(ratio * 38)

def standardizing_clean_sheets(row):
    clean_sheets = row["clean_sheets"]
    games = row["games"]
    ratio = clean_sheets/games
    return round(ratio * 38)

def standardizing_shots_on_target(row):
    shots_on_target = row["shots_on_target"]
    games = row["games"]
    ratio = shots_on_target/games
    return round(ratio * 38)

def standardizing_points(row):
    wins = row["wins"]
    draws = row["draws"]
    games = row["games"]
    ratio = (wins/games) * 38
    ratio1 = (draws/games) * 38
    return round((3 * ratio) + ratio1)

def standardizing_points_per_match(row):
    wins = row["wins"]
    draws = row["draws"]
    games = row["games"]
    ratio = (wins/games) * 38
    ratio1 = (draws/games) * 38
    result = (3 * ratio) + ratio1
    return round((result / 38), 2)

dataset["adjusted_wins"] = dataset.apply(standardizing_wins, axis = 1)
dataset["adjusted_draws"] = dataset.apply(standardizing_draws, axis = 1)
dataset["adjusted_losses"] = dataset.apply(standardizing_losses, axis = 1)
dataset["adjusted_goals_for"] = dataset.apply(standardizing_goals_for, axis = 1)
dataset["adjusted_goals_against"] = dataset.apply(standardizing_goals_against, axis = 1)
dataset["adjusted_goal_diff"] = dataset.apply(standardizing_goal_diff, axis = 1)
dataset["adjusted_assists"] = dataset.apply(standardizing_assists, axis = 1)
dataset["adjusted_pens_made"] = dataset.apply(standardizing_pens_made, axis = 1)
dataset["adjusted_pens_att"] = dataset.apply(standardizing_pens_att, axis = 1)
dataset["adjusted_shots_on_target_against"] = dataset.apply(standardizing_shots_on_target_against, axis = 1)
dataset["adjusted_saves"] = dataset.apply(standardizing_saves, axis = 1)
dataset["adjusted_clean_sheets"] = dataset.apply(standardizing_clean_sheets, axis = 1)
dataset["adjusted_shots_on_target"] = dataset.apply(standardizing_shots_on_target, axis = 1)
dataset["adjusted_points"] = dataset.apply(standardizing_points, axis = 1)
dataset["adjusted_points_per_match"] = dataset.apply(standardizing_points_per_match, axis = 1)

# Fixing Paris SG
dataset.replace({"Paris S-G": "Paris SG"}, inplace = True)

# Rank column to Categorical
dataset["rank"] = dataset["rank"].astype(object)

# Fixing the notes column and dropping when done
"""
ERRORS FOUND IN THE NOTES COLUMN TO BE CORRECTED WITH FUNCTIONS


# ---> Creating the first column UEFA Champions League
2010-2011 (England) - Manchester Utd, Chelsea, Arsenal, Manchester City
2011-2012 (England) - Manchester City, Manchester Utd, Arsenal, Chelsea
2012-2013 (England) - Manchester Utd, Chelsea, Arsenal, Manchester City
2013-2014 (England) - Manchester City, Liverpool, Chelsea, Arsenal
2014-2015 (England) - Manchester Utd, Chelsea, Arsenal, Manchester City
2015-2016 (England) - Leicester City, Arsenal, Tottenham, Manchester City
2016-2017 (England) - Chelsea, Tottenham, Manchester City, Liverpool, Manchester Utd
2017-2018 (England) - Manchester City, Manchester Utd, Tottenham, Liverpool
2018-2019 (England) - Manchester City, Liverpool, Chelsea, Tottenham
2019-2020 (England) - Liverpool, Manchester City, Manchester Utd, Chelsea
2020-2021 (England) - Manchester City, Manchester Utd, Liverpool, Chelsea

2020-2021 (La Liga) - Villarreal


# ---> Creating the second column Europa League
2010-2011 (England) - Tottenham, Stoke City, Fulham, Birmingham City
2011-2012 (England) - Tottenham, Newcastle Utd, Liverpool
2012-2013 (England) - Tottenham, Swansea City
2013-2014 (England) - Everton, Tottenham, Hull City
2014-2015 (England) - Tottenham, Liverpool, Southampton, West Ham
2015-2016 (England) - Manchester Utd, Southampton, West Ham
2016-2017 (England) - Arsenal, Everton
2017-2018 (England) - Chelsea, Arsenal, Burnley
2018-2019 (England) - Arsenal, Manchester Utd, Wolves
2019-2020 (England) - Leicester City, Arsenal, Tottenham
2020-2021 (England) - Leicester City, West Ham, Tottenham

2011-2012 (Ligue 1) - Lyon
2012-2013 (Ligue 1) - Saint-Étienne
2013-2014 (Ligue 1) - Lyon, Guingamp
2014-2015 (Ligue 1) - Marseille, Saint-Étienne
2015-2016 (Ligue 1) - Nice, Lille
2016-2017 (Ligue 1) - Lyon, Marseille
2017-2018 (Ligue 1) - Rennes, Bordeaux
2018-2019 (Ligue 1) - Rennes, Strasbourg
2020-2021 (Ligue 1) - Marseille

2011-2012 (Fußball-Bundesliga) - Leverkusen
2012-2013 (Fußball-Bundesliga) - Freiburg
2013-2014 (Fußball-Bundesliga) - Wolfsburg
2014-2015 (Fußball-Bundesliga) - Schalke 04
2015-2016 (Fußball-Bundesliga) - Mainz 05
2016-2017 (Fußball-Bundesliga) - Hertha BSC
2017-2018 (Fußball-Bundesliga) - Eint Frankfurt
2018-2019 (Fußball-Bundesliga) - Wolfsburg
2019-2020 (Fußball-Bundesliga) - Leverkusen
2020-2021 (Fußball-Bundesliga) - Leverkusen

2010-2011 (Serie A) - Roma
2011-2012 (Serie A) - Napoli
2013-2014 (Serie A) - Fiorentina
2014-2015 (Serie A) - Napoli
2015-2016 (Serie A) - Fiorentina
2016-2017 (Serie A) - Lazio
2017-2018 (Serie A) - Milan
2018-2019 (Serie A) - Lazio
2019-2020 (Serie A) - Napoli
2020-2021 (Serie A) - Lazio

2010-2011 (La Liga) - Athletic Club
2012-2013 (La Liga) - Valencia
2014-2015 (La Liga) - Villarreal
2015-2016 (La Liga) - Celta Vigo
2016-2017 (La Liga) - Real Sociedad
2017-2018 (La Liga) - Betis
2018-2019 (La Liga) - Sevilla
2019-2020 (La Liga) - Villarreal
2020-2021 (La Liga) - Betis


# ---> Creating the third column Relegation
2010-2011 (England) - Birmingham City, Blackpool, West Ham
2011-2012 (England) - Bolton, Blackburn, Wolves
2012-2013 (England) - Wigan Athletic, Reading, QPR
2013-2014 (England) - Norwich City, Fulham, Cardiff City
2014-2015 (England) - Hull City, Burnley, QPR
2015-2016 (England) - Newcastle Utd, Norwich City, Aston Villa
2016-2017 (England) - Hull City, Middlesbrough, Sunderland
2017-2018 (England) - Swansea City, Stoke City, West Brom
2018-2019 (England) - Cardiff City, Fulham, Huddersfield
2019-2020 (England) - Bournemouth, Watford, Norwich City
2020-2021 (England) - Fulham, West Brom, Sheffield Utd

2016-2017 (Ligue 1) - Lorient, Bastia
2017-2018 (Ligue 1) - Toulouse
2018-2019 (Ligue 1) - Dijon
2020-2021 (Ligue 1) - Nantes

2010-2011 (Fußball-Bundesliga) - M'Gladbach
2011-2012 (Fußball-Bundesliga) - Hertha BSC
2012-2013 (Fußball-Bundesliga) - Hoffenheim
2013-2014 (Fußball-Bundesliga) - Hamburger SV
2014-2015 (Fußball-Bundesliga) - Hamburger SV
2015-2016 (Fußball-Bundesliga) - Eint Frankfurt
2016-2017 (Fußball-Bundesliga) - Wolfsburg
2017-2018 (Fußball-Bundesliga) - Wolfsburg
2018-2019 (Fußball-Bundesliga) - Stuttgart
2019-2020 (Fußball-Bundesliga) - Werder Bremen
2020-2021 (Fußball-Bundesliga) - Köln

2011-2012 (Serie A) - Lecce
2012-2013 (Serie A) - Siena
2014-2015 (Serie A) - Parma
2018-2019 (Serie A) - Chievo

2014-2015 (La Liga) - Elche, Almería

---> SOURCE: FlashFootball - https://www.flashfootball.com/
"""
def champions_league(row):
    team = row["squad"]
    league = row["competition"]
    season = row["season"]
    if isinstance(row["notes"], str):
        if "Champions" in row["notes"].split():
            return "Yes"
        
    if season == "2010-2011" and league == "Premier League" and team in ["Manchester Utd", "Chelsea", "Arsenal", "Manchester City"]:
        return "Yes"
    elif season == "2011-2012" and league == "Premier League" and team in ["Manchester City", "Manchester Utd", "Arsenal", "Chelsea"]:     
        return "Yes"                           
    elif season == "2012-2013" and league == "Premier League" and team in ["Manchester Utd", 'Chelsea', "Arsenal", "Manchester City"]:     
        return "Yes"   
    elif season == "2013-2014" and league == "Premier League" and team in ["Manchester City", "Liverpool", "Chelsea", "Arsenal"]:     
        return "Yes"   
    elif season == "2014-2015" and league == "Premier League" and team in ["Manchester Utd", "Chelsea", "Arsenal", "Manchester City"]:     
        return "Yes"   
    elif season == "2015-2016" and league == "Premier League" and team in ["Leicester City", "Arsenal", "Tottenham", "Manchester City"]:     
        return "Yes"   
    elif season == "2016-2017" and league == "Premier League" and team in ["Chelsea", "Tottenham", "Manchester City", "Liverpool", "Manchester Utd"]:     
        return "Yes"   
    elif season == "2017-2018" and league == "Premier League" and team in ["Manchester City", "Manchester Utd", "Tottenham", "Liverpool"]:     
        return "Yes"   
    elif season == "2018-2019" and league == "Premier League" and team in ["Manchester City", "Liverpool", "Chelsea", "Tottenham"]:     
        return "Yes"   
    elif season == "2019-2020" and league == "Premier League" and team in ["Liverpool", "Manchester City", "Manchester Utd", "Chelsea"]:     
        return "Yes"   
    elif season == "2020-2021" and league == "Premier League" and team in ["Manchester City", "Manchester Utd", "Liverpool", "Chelsea"]:     
        return "Yes"   
    elif season == "2020-2021" and league == "La Liga" and team == "Villarreal":     
        return "Yes"
    else:
        return "No"



def europa_league(row):
    team = row["squad"]
    league = row["competition"]
    season = row["season"]
    if isinstance(row["notes"], str):
        if "Conference" in row["notes"].split():
            return "No"
        elif "Europa" in row["notes"].split():
            return "Yes"
        
    if season == "2010-2011" and league == "Premier League" and team in ["Tottenham", "Stoke City", "Fulham", "Birmingham City"]:     
        return "Yes"
    elif season == "2011-2012" and league == "Premier League" and team in ["Tottenham", "Newcastle Utd", "Liverpool"]:     
        return "Yes"
    elif season == "2012-2013" and league == "Premier League" and team in ["Tottenham", "Swansea City"]:     
        return "Yes"
    elif season == "2013-2014" and league == "Premier League" and team in ["Everton", "Tottenham", "Hull City"]:     
        return "Yes"
    elif season == "2014-2015" and league == "Premier League" and team in ["Tottenham", "Liverpool", "Southampton", "West Ham"]:     
        return "Yes"
    elif season == "2015-2016" and league == "Premier League" and team in ["Manchester Utd", "Southampton", "West Ham"]:     
        return "Yes"
    elif season == "2016-2017" and league == "Premier League" and team in ["Arsenal", "Everton"]:     
        return "Yes"
    elif season == "2017-2018" and league == "Premier League" and team in ["Chelsea", "Arsenal", "Burnley"]:     
        return "Yes"
    elif season == "2018-2019" and league == "Premier League" and team in ["Arsenal", "Manchester Utd", "Wolves"]:     
        return "Yes"
    elif season == "2019-2020" and league == "Premier League" and team in ["Leicester City", "Arsenal", "Tottenham"]:     
        return "Yes"
    elif season == "2020-2021" and league == "Premier League" and team in ["Leicester City", "West Ham", "Tottenham"]:     
        return "Yes"
    elif season == "2011-2012" and league == "Ligue 1" and team == "Lyon":     
        return "Yes"
    elif season == "2012-2013" and league == "Ligue 1" and team == "Saint-Étienne":     
        return "Yes"
    elif season == "2013-2014" and league == "Ligue 1" and team in ["Lyon", "Guingamp"]:     
        return "Yes"
    elif season == "2014-2015" and league == "Ligue 1" and team in ["Marseille", "Saint-Étienne"]:     
        return "Yes"
    elif season == "2015-2016" and league == "Ligue 1" and team in ["Nice", "Lille"]:     
        return "Yes"
    elif season == "2016-2017" and league == "Ligue 1" and team in ["Lyon", "Marseille"]:     
        return "Yes"
    elif season == "2017-2018" and league == "Ligue 1" and team in ["Rennes", "Bordeaux"]:     
        return "Yes"
    elif season == "2018-2019" and league == "Ligue 1" and team in ["Rennes", "Strasbourg"]:     
        return "Yes"
    elif season == "2020-2021" and league == "Ligue 1" and team == "Marseille":     
        return "Yes"
    elif season == "2011-2012" and league == "Fußball-Bundesliga" and team == "Leverkusen":     
        return "Yes"
    elif season == "2012-2013" and league == "Fußball-Bundesliga" and team == "Freiburg":     
        return "Yes"
    elif season == "2013-2014" and league == "Fußball-Bundesliga" and team == "Wolfsburg":     
        return "Yes"
    elif season == "2014-2015" and league == "Fußball-Bundesliga" and team == "Schalke 04":     
        return "Yes"
    elif season == "2015-2016" and league == "Fußball-Bundesliga" and team == "Mainz 05":     
        return "Yes"
    elif season == "2016-2017" and league == "Fußball-Bundesliga" and team == "Hertha BSC":     
        return "Yes"
    elif season == "2017-2018" and league == "Fußball-Bundesliga" and team == "Eint Frankfurt":     
        return "Yes"
    elif season == "2018-2019" and league == "Fußball-Bundesliga" and team == "Wolfsburg":     
        return "Yes"
    elif season == "2019-2020" and league == "Fußball-Bundesliga" and team == "Leverkusen":     
        return "Yes"
    elif season == "2020-2021" and league == "Fußball-Bundesliga" and team == "Leverkusen":     
        return "Yes"
    elif season == "2010-2011" and league == "Serie A" and team == "Roma":     
        return "Yes"
    elif season == "2011-2012" and league == "Serie A" and team == "Napoli":     
        return "Yes"
    elif season == "2013-2014" and league == "Serie A" and team == "Fiorentina":     
        return "Yes"
    elif season == "2014-2015" and league == "Serie A" and team == "Napoli":     
        return "Yes"
    elif season == "2015-2016" and league == "Serie A" and team == "Fiorentina":     
        return "Yes"
    elif season == "2016-2017" and league == "Serie A" and team == "Lazio":     
        return "Yes"
    elif season == "2017-2018" and league == "Serie A" and team == "Milan":     
        return "Yes"
    elif season == "2018-2019" and league == "Serie A" and team == "Lazio":     
        return "Yes"
    elif season == "2019-2020" and league == "Serie A" and team == "Napoli":     
        return "Yes"
    elif season == "2020-2021" and league == "Serie A" and team == "Lazio":     
        return "Yes"
    elif season == "2010-2011" and league == "La Liga" and team == "Athletic Club":     
        return "Yes"
    elif season == "2012-2013" and league == "La Liga" and team == "Valencia":     
        return "Yes"
    elif season == "2014-2015" and league == "La Liga" and team == "Villarreal":     
        return "Yes"
    elif season == "2015-2016" and league == "La Liga" and team == "Celta Vigo":     
        return "Yes"
    elif season == "2016-2017" and league == "La Liga" and team == "Real Sociedad":     
        return "Yes"
    elif season == "2017-2018" and league == "La Liga" and team == "Betis":     
        return "Yes"
    elif season == "2018-2019" and league == "La Liga" and team == "Sevilla":     
        return "Yes"
    elif season == "2019-2020" and league == "La Liga" and team == "Villarreal":     
        return "Yes"
    elif season == "2020-2021" and league == "La Liga" and team == "Betis":     
        return "Yes"
    else:
        return "No"
        
    
    
def relegation(row):
    team = row["squad"]
    league = row["competition"]
    season = row["season"]
    if isinstance(row["notes"], str):
        if "Relegated" in row["notes"].split():
            return "Yes"
    
    if season == "2010-2011" and league == "Premier League" and team in ["Birmingham City", "Blackpool", "West Ham"]:     
        return "Yes"
    elif season == "2011-2012" and league == "Premier League" and team in ["Bolton", "Blackburn", "Wolves"]:     
        return "Yes"
    elif season == "2012-2013" and league == "Premier League" and team in ["Wigan Athletic", "Reading", "QPR"]:     
        return "Yes"
    elif season == "2013-2014" and league == "Premier League" and team in ["Norwich City", "Fulham", "Cardiff City"]:     
        return "Yes"
    elif season == "2014-2015" and league == "Premier League" and team in ["Hull City", "Burnley", "QPR"]:     
        return "Yes"
    elif season == "2015-2016" and league == "Premier League" and team in ["Newcastle Utd", "Norwich City", "Aston Villa"]:     
        return "Yes"
    elif season == "2016-2017" and league == "Premier League" and team in ["Hull City", "Middlesbrough", "Sunderland"]:     
        return "Yes"
    elif season == "2017-2018" and league == "Premier League" and team in ["Swansea City", "Stoke City", "West Brom"]:     
        return "Yes"
    elif season == "2018-2019" and league == "Premier League" and team in ["Cardiff City", "Fulham", "Huddersfield"]:     
        return "Yes"
    elif season == "2019-2020" and league == "Premier League" and team in ["Bournemouth", "Watford", "Norwich City"]:     
        return "Yes"
    elif season == "2020-2021" and league == "Premier League" and team in ["Fulham", "West Brom", "Sheffield Utd"]:     
        return "Yes"
    elif season == "2016-2017" and league == "Ligue 1" and team in ["Lorient", "Bastia"]:     
        return "Yes"
    elif season == "2017-2018" and league == "Ligue 1" and team == "Toulouse":     
        return "Yes"
    elif season == "2018-2019" and league == "Ligue 1" and team == "Dijon":     
        return "Yes"
    elif season == "2020-2021" and league == "Ligue 1" and team == "Nantes":     
        return "Yes"
    elif season == "2010-2011" and league == "Fußball-Bundesliga" and team == "M'Gladbach":     
        return "Yes"
    elif season == "2011-2012" and league == "Fußball-Bundesliga" and team == "Hertha BSC":     
        return "Yes"
    elif season == "2012-2013" and league == "Fußball-Bundesliga" and team == "Hoffenheim":     
        return "Yes"
    elif season == "2013-2014" and league == "Fußball-Bundesliga" and team == "Hamburger SV":     
        return "Yes"
    elif season == "2014-2015" and league == "Fußball-Bundesliga" and team == "Hamburger SV":     
        return "Yes"
    elif season == "2015-2016" and league == "Fußball-Bundesliga" and team == "Eint Frankfurt":     
        return "Yes"
    elif season == "2016-2017" and league == "Fußball-Bundesliga" and team == "Wolfsburg":     
        return "Yes"
    elif season == "2017-2018" and league == "Fußball-Bundesliga" and team == "Wolfsburg":     
        return "Yes"
    elif season == "2018-2019" and league == "Fußball-Bundesliga" and team == "Stuttgart":     
        return "Yes"
    elif season == "2019-2020" and league == "Fußball-Bundesliga" and team == "Werder Bremen":     
        return "Yes"
    elif season == "2020-2021" and league == "Fußball-Bundesliga" and team == "Köln":     
        return "Yes"
    elif season == "2011-2012" and league == "Serie A" and team == "Lecce":     
        return "Yes"
    elif season == "2012-2013" and league == "Serie A" and team == "Siena":     
        return "Yes"
    elif season == "2014-2015" and league == "Serie A" and team == "Parma":     
        return "Yes"
    elif season == "2018-2019" and league == "Serie A" and team == "Chievo":     
        return "Yes"
    elif season == "2014-2015" and league == "La Liga" and team in ["Elche", "Almería"]:     
        return "Yes"
    else:
        return "No"
          
dataset["UEFA Champions League"] = dataset.apply(champions_league, axis = 1)  
dataset["UEFA Europa League"] = dataset.apply(europa_league, axis = 1) 
dataset["Relegation"] = dataset.apply(relegation, axis = 1)
dataset = dataset.drop("notes", axis = 1)

# Saving relevant table from EDA
data_eda["data_correlation_matrix"].to_csv("datasets/exploratory_data_analysis_tables/Correlation_Matrix_Table.csv", index = True)

# Saving clean dataset for visualization
dataset.to_csv("datasets/PreProcessed Dataset - Big 5 European football leagues teams stats.csv", index = True)
