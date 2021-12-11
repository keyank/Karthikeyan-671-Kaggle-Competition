import pandas as pd 
import numpy as np
from collections import Counter


def entropy(p):
    if (p == 0) or (p == 1): 
        return 0
    q = 1-p
    entropy = -p*np.log(p) - q*np.log(q)
    return entropy


def get_entropy_index_df(df, label_name):
    labels = np.array(df[label_name])
    p = np.mean(labels)
    return entropy(p) # Changed from gini_index(p) to entropy(p)


# def get_info_gain(data, feature, label):
    
#     entropy_initial = get_entropy_index_df(data, 'Decision')

#     left_tree  = data[data[feature] == 1]  # Left is all the data with the feature as 1 
#     right_tree = data[data[feature] == 0]  # Right is all the data with the feature as 0 

#     left_weight = len(left_tree)/len(data)
#     right_weight =  len(right_tree)/len(data)

#     assert left_weight + right_weight == 1

#     entropy_left = get_entropy_index_df(left_tree, 'Decision')
#     entropy_right = get_entropy_index_df(right_tree, 'Decision')

#     entropy_final = left_weight*entropy_left + right_weight*entropy_right

#     entropy_reduction = entropy_initial - entropy_final
    
#     stats = {
#         'entropy_reduction': entropy_reduction, 
#         'entropy_initial': entropy_initial, 
#         'left_weight':  left_weight, 
#         'right_weight': right_weight, 
#         'entropy_left':entropy_left, 
#         'entropy_right': entropy_right
#     }
    
#     return stats 
    
    
def get_info_gain(data, feature, label):
    
    entropy_initial = get_entropy_index_df(data, 'Decision')
    
    trees = []
    weights = []
    
    unique_vals = list(set(data[feature]))
    
    for val in unique_vals: 
        tree = data[data[feature] == val]
        weight = len(tree)/len(data)
        trees.append(tree)
        weights.append(weight)
        
    entropies = []
    entropy_final = 0 
    
    for tree, weight in zip(trees, weights): 
        entropy_tree = get_entropy_index_df(tree, 'Decision')
        entropies.append(entropy_tree)
        entropy_final = entropy_final + weight*entropy_tree

    entropy_reduction = entropy_initial - entropy_final
    
    stats = {
        'entropy_reduction': entropy_reduction, 
        'entropy_initial': entropy_initial, 
        'weights' : weights, 
        'entropies': entropies
    }
    
    return stats 
    
    
    

def clean_Driving_to(feature, drop_first=False, prune=True):
    feature = pd.Series(feature)
    feature = pd.get_dummies(feature, prefix='Driving_to:', drop_first=drop_first)
    return feature



def clean_Passanger(feature, drop_first=False,  prune=True):
    feature = pd.Series(feature)
    feature = pd.get_dummies(feature, prefix='Passanger:', drop_first=drop_first)
    return feature



def clean_Weather(feature, drop_first=False, prune=True):
    '''
    fature is converted to sunny or Not sunny
    '''
    feature = pd.Series(feature)
    if prune: 
        feature = feature.replace({'Sunny': 0, 'Rainy': 1, 'Snowy': 1})
    feature = pd.get_dummies(feature, prefix='Weather:', drop_first=drop_first)
    return feature



def clean_Temperature(feature, drop_first=False, prune=True): 
    '''
    Temperature can be skipped. Weather seems to be more important than temperature. 
    '''
    feature = pd.Series(feature)
    feature = pd.get_dummies(feature, prefix='Temperature:', drop_first=drop_first)
    return feature



def clean_Time(feature, drop_first=False, prune=True): 
    '''
    For early mornings or late nights, chances of accepting a coupon is less. Highest in the afternoon
    Two options: 
        1. Either keep it as same or change it. 
    '''
    feature = pd.Series(feature)
    if prune: 
        feature = feature.replace({'6PM': 'moderate', '7AM': 'bad', 
                               '2PM': 'best', '10PM': 'bad', 
                               '10AM': 'moderate'})
    feature = pd.get_dummies(feature, prefix='Time:', drop_first=drop_first)
    return feature



def clean_Coupon(feature, drop_first=False, prune=True): 
    '''
    5 different kinds of coupons. This seems to be an important feature
    '''
    feature = pd.Series(feature)
    feature = pd.get_dummies(feature, prefix='Coupon:', drop_first=drop_first)
    return feature



def clean_Coupon_validity(feature, drop_first=False, prune=True): 
    '''
    Chances of accepting are much higher if validity is 1d rather than 2hs 
    '''
    feature = pd.Series(feature)
    feature = pd.get_dummies(feature, prefix='Coupon_validity:', drop_first=drop_first)
    return feature


def clean_Gender(feature, drop_first=False, prune=True): 
    '''
    Males are more probable to accept a coupon
    '''
    feature = pd.Series(feature)
    feature = pd.get_dummies(feature, prefix='Gender:', drop_first=drop_first)
    return feature

def clean_Age(feature, drop_first=False, prune=True): 
    '''
    '''
    feature = pd.Series(feature)
    feature = pd.get_dummies(feature, prefix='Age:', drop_first=drop_first)
    return feature


def clean_Maritalstatus(feature, drop_first=False, prune=True): 
    '''
    '''
    feature = pd.Series(feature)
    feature = pd.get_dummies(feature, prefix='Maritalstatus:', drop_first=drop_first)
    return feature


def clean_Children(feature, drop_first=False, prune=True): 
    '''
    '''
    feature = pd.Series(feature)
    feature = pd.get_dummies(feature, prefix='Children:', drop_first=drop_first)
    return feature



def clean_Education(feature, drop_first=False, prune=True): 
    '''
    Highly Educated people are less likely to accept a coupon
    '''
    feature = pd.Series(feature)
    feature = pd.get_dummies(feature, prefix='Education:', drop_first=drop_first)
    return feature



def clean_Occupation(feature, drop_first=False, prune=True): 
    '''
    Weather a person accepts coupons depends on their occupation. 
    '''
    feature = pd.Series(feature)
    feature = pd.get_dummies(feature, prefix='Occupation:', drop_first=drop_first)
    return feature


def clean_Income(feature, drop_first=False, prune=True): 
    '''
    Weather a person accepts coupons depends on their Income. 
    Such a small statification of income may lead to overfitting. 
    '''
    feature = pd.Series(feature)
    if prune: 
        feature = feature.replace({'$100000 or More': 'high',
                                 '$62500 - $74999': 'm-high',
                                 '$37500 - $49999': 'm-low',
                                 '$12500 - $24999': 'low',
                                 '$25000 - $37499': 'm-low',
                                 '$75000 - $87499': 'm-high',
                                 '$50000 - $62499': 'm-high',
                                 '$87500 - $99999': 'high',
                                 'Less than $12500': 'low'})
    feature = pd.get_dummies(feature, prefix='Income:', drop_first=drop_first)
    return feature


def clean_Bar(feature, drop_first=False, prune=True): 
    '''
    '''
    feature = pd.Series(feature)
    feature = feature.replace({np.nan: 0.0})
    feature = pd.get_dummies(feature, prefix='Bar:', drop_first=drop_first)
    return feature


def clean_Coffeehouse(feature, drop_first=False, prune=True): 
    '''
    '''
    feature = pd.Series(feature)
    feature = feature.replace({np.nan: 0.0})
    feature = pd.get_dummies(feature, prefix='Coffeehouse:', drop_first=drop_first)
    return feature



def clean_Carryaway(feature, drop_first=False, prune=True): 
    '''
    '''
    feature = pd.Series(feature)
    feature = feature.replace({np.nan: 0.0})
    feature = pd.get_dummies(feature, prefix='Carryaway:', drop_first=drop_first)
    return feature


def clean_Restaurantlessthan20(feature, drop_first=False, prune=True): 
    '''
    '''
    feature = pd.Series(feature)
    feature = feature.replace({np.nan: 0.0})
    feature = pd.get_dummies(feature, prefix='Restaurantlessthan20:', drop_first=drop_first)
    return feature



def clean_Restaurant20to50(feature, drop_first=False, prune=True): 
    '''
    '''
    feature = pd.Series(feature)
    feature = feature.replace({np.nan: 0.0})
    feature = pd.get_dummies(feature, prefix='Restaurant20to50:', drop_first=drop_first)
    return feature


def clean_Direction_same(feature, drop_first=False, prune=True): 
    '''
    '''
    feature = pd.Series(feature)
    return feature


def clean_Distance(feature, drop_first=False, prune=True): 
    '''
    '''
    feature = pd.Series(feature)
    feature = pd.get_dummies(feature, prefix='Distance:', drop_first=drop_first)
    return feature



def clean_all(data, drop_first, prune):

    Driving_to = clean_Driving_to(data['Driving_to'], drop_first=drop_first, prune=prune)
    Passanger = clean_Passanger(data['Passanger'], drop_first=drop_first, prune=prune)
    Weather = clean_Weather(data['Weather'], drop_first=drop_first, prune=prune)
    Temperature = clean_Temperature(data['Temperature'], drop_first=drop_first, prune=prune)
    Time = clean_Time(data['Time'], drop_first=drop_first, prune=prune)
    Coupon = clean_Coupon(data['Coupon'], drop_first=drop_first, prune=prune)
    Coupon_validity = clean_Coupon_validity(data['Coupon_validity'], drop_first=drop_first, prune=prune)
    Gender = clean_Gender(data['Gender'], drop_first=drop_first, prune=prune)
    Age = clean_Age(data['Age'], drop_first=drop_first, prune=prune)
    Maritalstatus = clean_Maritalstatus(data['Maritalstatus'], drop_first=drop_first, prune=prune)
    Children = clean_Children(data['Children'], drop_first=drop_first, prune=prune)
    Education = clean_Education(data['Education'], drop_first=drop_first, prune=prune)
    Occupation = clean_Occupation(data['Occupation'], drop_first=drop_first, prune=prune)
    Income = clean_Income(data['Income'], drop_first=drop_first, prune=prune)
    Bar = clean_Bar(data['Bar'], drop_first=drop_first, prune=prune)
    Coffeehouse = clean_Coffeehouse(data['Coffeehouse'], drop_first=drop_first, prune=prune)
    Carryaway = clean_Carryaway(data['Carryaway'], drop_first=drop_first, prune=prune)
    Restaurantlessthan20 = clean_Restaurantlessthan20(data['Restaurantlessthan20'], drop_first=drop_first, prune=prune)
    Restaurant20to50 = clean_Restaurant20to50(data['Restaurant20to50'], drop_first=drop_first, prune=prune)
    Direction_same = clean_Direction_same(data['Direction_same'], drop_first=drop_first, prune=prune)
    Distance = clean_Distance(data['Distance'], drop_first=drop_first, prune=prune)
    
    X_df = pd.concat([Driving_to, Passanger, Weather, Temperature,
       Time, Coupon, Coupon_validity, Gender, Age, Maritalstatus,
       Children, Education, Occupation, Income, Bar, Coffeehouse,
       Carryaway, Restaurantlessthan20, Restaurant20to50,
       Direction_same, Distance], axis=1)
    
    try: 
        y_df = data['Decision']
    except: 
        y_df = None 
        
    
    return X_df, y_df

