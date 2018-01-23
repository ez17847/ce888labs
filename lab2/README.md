# Lab2

## What was done?

##Histogram and Scaterplot
1. The file vehicles.csv was read into the variable df.
2. Using the values of the two columns, the scaterplots were made. 
![scaterplot](./scaterplot_fleet.png?raw=true)
3. The numpy arrays data1 and data0 were created. data0 containing the first column of df and data1 containing the second column of df with out its NA values.
4. The the mean, median, variance, standard deviation and MAD values for each list are calculated and shown.
5. Using the values of data1, the histogram is created.
![histogram](./histogram_fleet.png?raw=true)

##Standard deviation comparison via the boostrap	
1. Using the information in data0, the function provided for bootstrap is used.
2. Using iterations from 100 to 100000 (with intervals of 1000), upper and lower bounds of the standard deviation (and the mean) are obtained.
![boostrap](./bootstrap_confidence_Current.png?raw=true)
3. Using the same procedure for data1, upper and lower bounds of the standard deviation (and the mean) are obtained.
![boostrap](./bootstrap_confidence_Proposed.png?raw=true)

- [ ] Are the standard deviations comparable?
Yes, they are comparable. This is because any of the boundaries do not touch each other; thus, the limits of the current are not inside of the proposed, and vice versa.

