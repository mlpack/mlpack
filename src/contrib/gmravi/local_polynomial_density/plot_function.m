% This code plots the function w.r.t independent coordinates

function[]= PlotFunction()

x_vector=[];
y_vector=[];
% Read x_vector
x_vector=dlmread('points_mog13_small.txt');

%Read y_vector
y_vector=dlmread('density_points_mog13_small.txt');
length(x_vector)
length(y_vector)
plot(x_vector,y_vector,'.');
