# Project 2 - Arctic ice maps
## Aim : To become familiar with working on image data, plotting it, and combining it in various ways for analysis.


### Steps Involved:
### 1. Loading of data:
The data used in this problem set was/is collected by two different satellite missions: the AMSR-E instrument on the [Aqua](http://en.wikipedia.org/wiki/Aqua_%28satellite%29) satellite (data from 2002 to 2011) and the AMSR2 instrument on the [GCOM-W](https://suzaku.eorc.jaxa.jp/GCOM_W/) satellite (data from 2013 to-date).

The data consist of maps of the concentration of ice in the Arctic collected between 2002 and 2019 with the exception of 2012. All the data were already downloaded and transformed for you to an easy-to-use format from [here](https://seaice.uni-bremen.de/start/data-archive/).

The data we use are in the directory `/home/shared/Project_2/ice_data` within our online-system. This is actually a (small) subset of the complete satellite data set, with only two ice maps every month (some are missing though).
The data is in `numpy` format, which means that it can be read as a `numpy`-array.

### 2. Analysis Task : 
Since we successfully load the data, the next steps involve the analysis and plotting of the data.

#### Task 1: Getting familiar with the data and examining a single map 
In this part we discuss the following things:
   * We start off by reading in some  map, and plot it with Matplotlib.
   * To plot a colorbar on the side, to show what color corresponds to what value.

    "**Remarks:**
    (1) When you explore the data-values, you will notice that they contain numbers from 0 to 100. A value of 50 means that 50% of the area occupied by the corresponding pixel are covered with ice. A value of zero means that the complete pixel is covered with water; 
    (2) Besides the numbers, a good deal of the pixels contains the special value `nan` (not a number). These are areas covered by land.
       Functions like `imshow` automatically ignore these values and do not produce an error.]
 
  
  #### Task 2: Reading in and examining multiple maps
  In this part we discuss the following things:
   * Plot of the ice concentration over time.
   * We compute for each file the total number of pixels that have a value above 50% ice.
   * Plot of the number of pixels with a concentration above 50% against time.
   * Ivestigate how the pixel area is changing over the image.
   * We then compute the total area where the concentration of ice is 99% or above and make a new plot showing the area of >99% ice concentration against time. 
   * Lastly, we plot the *total area* covered by ice as a function of time.
  #### Task 3: Visualizing changes over time; does the amount of ice decrease?
   * Find the date at which the area of the region covered with ice is the smallest. What is the value of the minimum area?
   * How does the minimum value within each year change over time?
   * Find the date at which the area of the region covered with ice is the smallest. What is the value of the minimum area?
   * Compute the difference between the two maps so that a loss in ice over time will correspond to a negative value, and a gain in ice will correspond to a positive value. Make a plot of the difference, and use the ``RdBu`` colormap to highlight the changes (include a colorbar).
   * Finally, plot the `mean` ice concentrations of the years 2004-2006 over the months January to December and compare it to the mean over the years 2017-2019. What was the loss of the ice over the past 15 years at the minimum values?

  
