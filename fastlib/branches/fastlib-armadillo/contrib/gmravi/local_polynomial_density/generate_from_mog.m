function[]=generate_from_mog()
data=dlmread('weights_13.txt',',');
%Number of points to be generated
number_of_points=6000;
display('Number of points to be generated is 6K');
[rows,cols]=size(data);
number_of_components=rows;
numbers=[];

%Randomly seed
rand('state',sum(100*clock));
for i=1:number_of_points
    %First generate a random number
    component_num=ceil(rand(1)*number_of_components);
 
    %Now generate a number from this component
    mean=data(component_num,2);
    sigma=sqrt(data(component_num,3));
    numbers(i)=mean+sigma*randn(1);
end 

density=GetDensity(numbers,data);
PrintToFile(numbers,density);

function [densities]= GetDensity(numbers,data)
%Get weights of the components and their means and sigma
[rows,cols]=size(data);
number_of_components=rows;
weights_of_components=data(1:end,1:1);
means=data(1:end,2:2);
sigma_sqd=data(1:end,3:3);
%For each point find the density

density=0;
for i=1:length(numbers) %For each point
    density=0;
    %display('number generated is');
    %display(numbers(i));
    for j=1:number_of_components % For each component
        density=density+weights_of_components(j)*kg(numbers(i),means(j),sigma_sqd(j));
        %display('density becomes');
        %display(density);
    end
    densities(i)=density;
end


function k=kg(x,mean,sigma_sqd)
k=exp(-0.5*(x-mean)*(x-mean)/sigma_sqd)/sqrt(2*pi*sigma_sqd);
    
function  PrintToFile(numbers,density)

%print numbers and their densities
[n_rows,n_cols]=size(numbers);
display(n_rows);
display(n_cols);

%dlmwrite('/net/hc295/gmravi/home/fastlib/fastlib2_int/fastlib2/contrib/gmravi/density_estimation/points_mog1_train1.txt',numbers(1:ceil(n_cols/2)),'\n');
dlmwrite('points_mog13_small.txt',numbers(1:2*n_cols/3),'\n');
%Print the densities

dlmwrite('density_points_mog13_small.txt',density(1:2*n_cols/3),'\n');

dlmwrite('points_mog13_small_test.txt',numbers(2*n_cols/3+1),'\n');
%Print the densities

dlmwrite('density_points_mog13_small_test.txt',density(2*n_cols/3+1:end),'\n');




    

    
    
    
    