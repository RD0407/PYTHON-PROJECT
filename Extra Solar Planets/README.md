# Project 4 - Extrasolar Planets
## Aim : to investigate a dataset that contains the radial velocity of a star at various points in time.
## Steps Involved:
This project consists of two parts:

In the first part, we learn about a Monte-Carlo method to estimate errors from a quantity that is compound of other quantities with measurement errors.
 
This method is then used to estimate errors on the mass from the stars companion in the second part.

### 2. Analysis Task
Analysis and plotting of the data:

#### Task 1: Monte-Carlo Error Propagation
   * Use the [standard error propagation rules](http://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulas) to determine the resulting force and uncertainty in a `python`-program
   * We also try using a **Monte-Carlo** technique.
   * We make a comparison as of which method is more accurate? 

#### Task 2: Analysis of a Extrasolar Planet candidate
The data file required for this project is [data/UID_0113357_RVC_001.tbl](data/UID_0113357_RVC_001.tbl). It was obtained from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/DisplayOverview/nph-DisplayOverview?objname=51+Peg+b&type=CONFIRMED_PLANET).
   * We obtain the file, and then carry out the analysis described in Tasks 1.
   * We then determine the radial velocity of the host star.
   ##### Strategy : 
   We want to see whether the star does indeed show periodic variations, and if so, we want to measure the period and amplitude. The amplitude is then a direct measure of the radial velocity.
   
   #### Task 3 : Mass analysis of the extrasolar planet 51 Peg b
   
   * We try to estimate the mass of the invisible companion that is orbiting the star. 
   * We then derive the mass of the object in units of the mass of Jupiter.
   * We then plot a histogram of the probability that the object has a certain mass, and show only the range from 0 to 13 times the mass of Jupiter.
   * We then measure what degree of confidence do we have that the object is a planet, using the 1/2/3/4/5-sigma confidence terminology...
   
    
     
      
      
      
