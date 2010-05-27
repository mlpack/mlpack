function[]=PluginBandwidth()
%This code accepts a data file and calculates the plugin bandwidth for that
%dataset

data=dlmread('/net/hc295/gmravi/home/fastlib/fastlib/u/gmravi/local_polynomial_density/points_mog1_train1.txt',',');

[rows,cols]=size(data);
number_of_points=rows;
%get standard-deviation
stand_dev=std(data);

%Get interquartile range
interquartile_range=iqr(data);

%The plugin bandwidth is 
display('number of points is ');
display(number_of_points');
plugin_bw=0.9*min(stand_dev,interquartile_range/1.34)*number_of_points^-0.2;
display('plugin bw is');
display(plugin_bw);