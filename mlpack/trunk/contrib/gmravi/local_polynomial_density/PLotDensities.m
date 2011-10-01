function[]=PlotDensities()
str='results_density_13.txt';
data=dlmread(str);
%Having read all data get to plot them 
points=data(1:end,1:1);
local_polynomial=data(1:end,2:2);
local_likelihood=data(1:end,3:3);
naive_kde=data(1:end,4:4);
true_density=data(1:end,5:5);
%local_likelihood_densities=densities(4:4);
[number_of_rows,]=size(data);

%plot_matrix(1:number_of_rows,1:1)=points;
plot_matrix(1:number_of_rows,1:1)=local_polynomial;

plot_matrix(1:number_of_rows,2:2)=local_likelihood;
plot_matrix(1:number_of_rows,3:3)=naive_kde;
plot_matrix(1:number_of_rows,4:4)=true_density;
plot(points,plot_matrix,'.');


legend('local-polynomial','local-likelihood','naive-kde','true-density');
title(str);

%calculate an erf value

%b1=erf(sqrt(3.0/8.0)*1.5)-erf(sqrt(3.0/8.0)*0.5);
%b2=erf(sqrt(3.0/8.0)*2.5)-erf(sqrt(3.0/8.0)*1.5);
%b1*b1
%b1*b2
%b2*b1
%b2*b2
