## Brief intro ## 
This is the Readme file for the second derivative fitting routine developed by Shrividya Ravi, VUW, Wellington, New Zealand. 

## History ## 
This script was developed to fit Raman spectra from heavily bundled single walled carbon nanotubes (in thin films). Since the spectral components change with different spot on the sample, the script required an interactive component - where the user can guide the fitting routine. This interactivitiy part was obtained from: http://scienceoss.com/interactively-select-points-from-a-plot-in-matplotlib/

## The script ## 
This script provides a template for anypne wanting to perform high resolution fitting on heavily convoluted spectra. The script does the following:

- smooth experimental data according to:
    - Savitzky-Golay (http://www.scipy.org/Cookbook/SavitzkyGolay)
    - moving average smoothing (triangular window)
- calculates second derivative of data

- plots second derivative of data and asks user to click on N peaks

- The user can move to the next set of data with a right click of the mouse or close the window for the script to commence fitting

- the user input is taken as part of the set of initialisation parameters for fitting

- The second derivative spectrum is fitted according the number of peaks chosen by the user.

- The script finishes by showing the spectral data and the second derivative of the data and the optimised fit to this data.

## To come ## 
The above workflow was only part of the full fitting routine. The remaining part of the routine included two rounds of additional fitting to the original spectrum - using the optimised parameters in the second derivative fitting. These additional fitting rounds allowed the peak intensities to be fitted accurately, and for the user to discard/adjust any unphysical fitted parameters. 

## Additional resources ## 
The accompanying pdf (selected pages from my PhD thesis) illustrates the second derivative fitting with figures. 

