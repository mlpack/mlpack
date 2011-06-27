function [numbers density]=generate_from_mog(number_of_points)
data=csvread('weights_3.txt');
%Number of points to be generated
%number_of_points=300;
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
data_filename = sprintf('spiky_mog_%d.csv', number_of_points);
density_filename = sprintf('density_spiky_mog_%d.csv', number_of_points);
data_filename
csvwrite(data_filename, numbers');
csvwrite(density_filename, density');
%PrintToFile(numbers,density);


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
        density=density+...
            weights_of_components(j)*kg(numbers(i),means(j),sigma_sqd(j));
        %display('density becomes');
        %display(density);
    end
    densities(i)=density;
end


function k=kg(x,mean,sigma_sqd)
k=exp(-0.5*(x-mean)*(x-mean)/sigma_sqd)/sqrt(2*pi*sigma_sqd);
    
function  PrintToFile(numbers,density,frac)

%print numbers and their densities
[n_rows,n_cols]=size(numbers);
display(n_cols);
display(n_rows);

%Train points
%display('Number of train points are');
%display(frac*n_cols);

%Write train and test points
%dlmwrite('/net/hu17/gmravi/fastlib2/contrib/gmravi/cake/qp_and_boosted_kde/mog3_qp_kde/points_mog3_train_1600.txt',numbers(1:frac*n_cols),'\n');

dlmwrite('/net/hu17/gmravi/fastlib2/contrib/gmravi/cake/qp_and_boosted_kde/mog3_qp_kde/points_mog3_test_1600.txt',...
        numbers(frac*n_cols+1:end),'\n');
    
dlmwrite('/net/hu17/gmravi/fastlib2/contrib/gmravi/cake/qp_and_boosted_kde/mog3_qp_kde/density_points_mog3_test_1600.txt',...
       density(frac*n_cols+1:end),'\n');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%dlmwrite('/net/hu17/gmravi/fastlib2/contrib/gmravi/hyperkernel_kde/mog3/points_mog3_train.txt',numbers(1:frac*n_cols),'\n');
 %   
%dlmwrite('/net/hu17/gmravi/fastlib2/contrib/gmravi/hyperkernel_kde/mog3/points_mog3_test.txt',...
 %       numbers(frac*n_cols+1:end),'\n');
    
%dlmwrite('/net/hu17/gmravi/fastlib2/contrib/gmravi/hyperkernel_kde/mog3/density_points_mog3_test.txt',...
 %      density(frac*n_cols+1:end),'\n');    
