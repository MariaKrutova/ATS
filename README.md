# ATS
Adaptive Threshold Segmentation (ATS) method for the small shapes detection on a uniform background.

The code is written for Python 3 and requires the following libraries:
- NumPy
- SciPy        (for fitting and splitting thresholded image into parts)
- Matplotlib   (only for the plots)
- scikit-image (to load image data)

The method and its application was decribed for the whitecaps detection in [1] Bakhoday-Paskyabi, M., Reuder, J. and Flügge, M.: Automated measurements of whitecaps on the ocean surface from a buoy-mounted camera, Methods Oceanogr., 17, 14–31, doi:10.1016/j.mio.2016.05.002, 2016.

and for the wake identification in [2] Krutova, M., Bakhoday-Paskyabi, M., Reuder, J. and Nielsen, F. G.: Development of an automatic thresholding method for wake meandering studies and its application to the data set from scanning wind lidar, Wind Energy Sci., 7(2), 849–873, doi:10.5194/wes-7-849-2022, 2022.

The methoud is suitable for identifying small structures on a predominantly uniform background. I.e., if the values contained in the image/data matrix are plotted as a histogram, the peak is expected to correspond to the background, while the structure in question is concentrated in the tail. If the histogram of values distribution tends to double or more prominent peaks, this method may not return the expected results.

Provided with the code are examples of the wind turbine wakes and the photo of the sea surface with a whitecap. The wake data are provided both as image and radial velocity values from the lidar.


.\examples
  |- lidar_data.npy     (lidar data stored in polar coordinates as a 3D matrix)
  |- lidar_img.png      (lidar data plotted as a grascaly image in the Cartesian coordinates)
  |- netcdf_img.png     (NetCDF data of a large-eddy simulation stored as a grayscale image)
  |- whitecap_img.png   (sea surface photo with a whitecap used in [1] as an example)
helper_functions.py     (small functions not related to the image processing)
main_ATS.py             (main function)
processing_functions.py (ATS functions)
