import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

# The following code produces some warnings that mostly confuse
# people. Hence, we disable them for now.
import warnings
warnings.filterwarnings('ignore')

# The Web-address where we can obtain daily updated Covid-19 data
corona_data_source = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv/'

class CoronaData(object):
    """
    A class to retrieve and to make availabe Covid-19 data from
    the 'European Centre for Disease Prevention and Control'. The data
    are obtained from
    https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide
    """
    
    def __init__(self, datafile=None, day=None):
        """
        The Covid-19 data can be populated online or from a file that was
        already downloaded
        """
        
        # available countries 
        self.countries = []
        
        data_source = corona_data_source
        self.covid_day = dt.date.today()
        if datafile != None:
            data_source = datafile
            self.covid_day = dt.date(*(day))
            
        self.covid_data = pd.read_csv(data_source, engine="python")

        self.countries = list(set(self.covid_data['countriesAndTerritories']))
        self.countries.sort()
        
    def __len__(self):
        """
        return the number of available countries
        """
        
        return len(self.countries)
    
    def __getitem__(self, country):
        """
        return a tuple of 'day', 'new infections' and 'deaths' per day
        for a given country. The 0th element (day 0) represents the
        first of March 2020 (Day 0 of the Covid-19 pandemy)
        """
        
        country_data = self.covid_data[
            self.covid_data['countriesAndTerritories'] == country]

        # The Covid-19 data contain data from the 31st of December 2019
        # onwards but NOT for all countries. We only want to consider data
        # from the first of March and hence we remove unnecessary entries.
        # Furthermore, we revert the
        # order of the data so that data for the the 1st of March is
        # at the first array position, not at the last:
        
        # we get the data we need by calculating the number of days that passed
        # sind the first of March:
        day0 = dt.date(2020, 3, 1)
        today = self.covid_day
        dayX = (today - day0).days
        
        if len(country_data['cases']) < dayX:
            days = np.arange(dayX + 1 - len(country_data['cases']), dayX + 1, 1)
            cases = np.array(country_data['cases'][::-1])
            deaths = np.array(country_data['deaths'][::-1])
        else:
            days = np.arange(0, dayX + 1, 1)
            cases = np.array(country_data['cases'][:(dayX + 1)][::-1])
            deaths = np.array(country_data['deaths'][:(dayX + 1)][::-1])
            
        return days, cases, deaths    
    
