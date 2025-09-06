# Project 1 - COVID-19 Data Analysis
## Aim: To analyse the development of the COVID-19 pandemy and to check which countries currently do have a raising number of infectious COVID-19 patients. 
## Steps Involved:
### 1. Loading of data:
The data used in this notebook are daily updated and published by the [European Centre for Disease Prevention and Control](https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide).

The  sequence of commands to retrieve interesting time-sequence data for specific countries have been transferred to the module `corona_data`. It reads the data and extracts the columns *cases* and *deaths* for a specific country. Furthermore, it removes all data before the 1st of March 2020. This date, we consider our *Day Zero* of the pandemy henceforth.

### 2. Analysis Task
Since we successfully load the data, the next steps involve the analysis and plotting of the data.
#### Plot 1: The *total accumulated number* of COVID-19 cases against the day for Germany.
In this part we discuss the following things:
   * What kind of curve does one expect for a pandemy that can spread freely? 
   * What effect do the measures and restrictions in Germany (e.g. social distancing) have on the curve? 
   * With this knowledge, a discussion has been made on that drastic limitations on our life (closure of schools etc.) that took effect in Germany on the 16th of March, 2019.
   * What will the curve look like when the pandemy is over ?
#### Plot 2: The infectious population as a function of pandemy-day for Germany.
In this part we discuss the following things:
   * Effect of COVID-19 restrictions on the infection rate.
   * Infection rate when the restrictions are removed and how long for this rate to reach its maximum.
#### Plot 3: The infectious population as a function of pandemy-day for all countries.
In this part we discuss the following things:-
   * Make a list of those countries which currently still have a raising infectious population. Only countries having more than 5000 confirmed cases are considered. 



