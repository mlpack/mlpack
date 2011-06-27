function[density_estimate_vec,contrib_mat,train_data,test_data]=local_rodeo(train_data,test_data,...
						  train_labels,test_labels,positive_prior,negative_prior)


%We shall do local rodeo. here for each test point we shall calculate
%bandwidths in different directions.

%The first thing to do is to rescale the data

[num_train_points]=size(train_data,1);

if (nargin < 2)
  
  %We are doing loo density estimation
  density_estimate_vec=ones(num_train_points,1);
  contrib_mat=[];
  for i=1:num_train_points
    test_point=train_data(i,:);
    loo_train_set = train_data([1:i-1, i+1:num_train_points], :);
    [vec_bandwidths]=CalculateBandwidths(test_point,loo_train_set);
    [density_estimate,contrib_vec]=CalculateDensity(vec_bandwidths,test_point,loo_train_set);
    contrib_mat=[contrib_mat;contrib_vec];
    density_estimate_vec(i)=density_estimate;
  end
  
else 

  [num_test_points]=size(test_data,1);

  min_train_data=min(train_data);
  max_train_data=max(train_data);

  num_dims=size(train_data,2);
  %display(num_dims);
  min_test_data=min(test_data);
  max_test_data=max(test_data);

  % display(min_test_data);
  %display(num_test_points);
  %display(num_train_points);
  boundary_train=[min_train_data,max_train_data];
  boundary_test=[min_test_data,max_test_data];

  boxed_train_data=[];
  boxed_test_data=[];

  if(num_dims< 1)  
    
    for j=1:num_dims
      col=...
	  (train_data(1:end,j:j)-repmat(min_train_data(j),num_train_points,1))/(max_train_data(j)-min_train_data(j));
      boxed_train_data=[boxed_train_data,col];
      col=...
	  (test_data(1:end,j:j)-repmat(min_test_data(j),num_test_points,1))/(max_test_data(j)-min_test_data(j));
      boxed_test_data=[boxed_test_data,col];
    end
    train_data=boxed_train_data;
    test_data=boxed_test_data;
  end


  if(nargin==2)
    %We are doing density estimation
    density_estimate_vec=ones(num_test_points,1);
    contrib_mat=[];
    for i=1:num_test_points
      test_point=test_data(i,:);
      [vec_bandwidths]=CalculateBandwidths(test_point,train_data);
      [density_estimate,contrib_vec]=CalculateDensity(vec_bandwidths,test_point,train_data);
      contrib_mat=[contrib_mat;contrib_vec];
      density_estimate_vec(i)=density_estimate;
    end
  else
    %WE are doing classification
    
    contrib_mat=[];
    num_correctly_classified=0;
    for i=1:num_test_points
      test_point=test_data(i:i,1:end);
      [vec_bandwidths]=CalculateBandwidths(test_point,train_data);
      
      [classification]=...
	  CalculateDensityForClassification(vec_bandwidths,...
					    test_point,train_data,train_labels,test_labels(i),...
					    positive_prior,negative_prior);
      num_correctly_classified=...
	  num_correctly_classified+classification;
    end
    display(num_correctly_classified);
    accuracy=num_correctly_classified/num_test_points;
    display('Accuracy is.............');
    display(accuracy)
  end

end


function[vector_bw]=CalculateBandwidths(test_point,train_data)
[num_dims]=size(test_point,2);
[num_train_points]=size(train_data,1);
c_0=1;
c_n=log(num_train_points);
beta=0.9;
h_0=...
    c_0/log(log(num_train_points));

zero_range_ind = [];

if(num_dims==1)
  %In this case we wont box the data. Hence multiply h_0 with the range
  %of the data
  min_data=min(train_data);
  max_data=max(train_data);
  range1=max_data-min_data;
  vector_bw=h_0*range1*ones(num_dims,1);
else
  vector_bw=h_0*range(train_data);
  zero_range_ind = find(range(train_data) == 0);
  vector_bw(zero_range_ind) = 1e-4;
end

ones_vec=ones(num_train_points,1);
active_dimensions= setdiff([1:num_dims], zero_range_ind);
% display(length(zero_range_ind));
% display(length(active_dimensions));
while(length(active_dimensions)~=0)
  
  iter=1;
  while(iter<=length(active_dimensions))
    dimension_in_consideration=active_dimensions(iter);
    [vector]=GetVectorAlongDirection(test_point,train_data,...
				     vector_bw,dimension_in_consideration);
    Z_j=(ones_vec'*vector)/num_train_points;
    s_j_sqd=var(vector)/num_train_points;
    lambda_j=sqrt(s_j_sqd)*sqrt(2*log(num_train_points*c_n));
    if(abs(Z_j)>lambda_j)
      vector_bw(dimension_in_consideration)=...
	  beta*vector_bw(dimension_in_consideration);
      if vector_bw(dimension_in_consideration) < 1e-4
	active_dimensions(iter) = [];
      end
      
    else
      
      %This dimension is no longer active
      active_dimensions(iter)=[];
    end
    iter=iter+1;
  end
  %display('Broke....');
  %display(active_dimensions);
  %display('Length of active dimensions is...');
  %display(length(active_dimensions));
end

function[return_vec]=...
    GetVectorAlongDirection(test_point,train_data,vector_bw,dir)

%Here we have been given a direction
test_point_coor=test_point(dir);
[num_train_points,num_dims]=size(train_data);
return_vec=ones(num_train_points,1);
bw=vector_bw(dir);

for i=1:num_train_points
  %get (x_j-X_{i,j})^2-h_j^2
  
  val=test_point_coor-train_data(i,dir);
  sum=0;
  
  for k=1:num_dims
    test_point_in_dir=test_point(k);
    train_point_in_dir=train_data(i,k);
    sum=sum+((test_point_in_dir-train_point_in_dir)/(sqrt(2)*vector_bw(k)))^2;
  end
  constant=exp(-sum);
  return_vec(i,1)=(val^2-bw^2)*constant;
end


function[density,contrib_vec]=CalculateDensity(vec_bandwidths,test_point,train_data)
contrib_vec=[];
[num_train_points,num_dims]=size(train_data);

total_contrib=0;
for i=1:num_train_points
  contrib=...
      CalculateContributionDueToTrainPoint(train_data(i:i,1:end),...
					   test_point,vec_bandwidths);
  total_contrib=total_contrib+contrib;                              
  contrib_vec=[contrib_vec,contrib];
end
density=total_contrib/num_train_points;

function[correctly_classified]=...
    CalculateDensityForClassification(vec_bandwidths,test_point,train_data,...
				      train_labels,test_label,positive_prior,negative_prior)

[num_train_points,num_dims]=size(train_data);

total_contrib=0;
for i=1:num_train_points
  contrib=...
      CalculateContributionDueToTrainPoint(train_data(i:i,1:end),...
					   test_point,vec_bandwidths);
  if(train_labels(i)==1)
    total_contrib=total_contrib+(contrib*positive_prior);
  else
    total_contrib=total_contrib-(contrib*negative_prior);
  end
end
if(sign(total_contrib*test_label)==1)

  correctly_classified=1;
  
else
  correctly_classified=0;
end

function[prod]=...
    CalculateContributionDueToTrainPoint(train_point,test_point,vec_bandwidths)

num_dims=size(train_point,2);
prod=1;
for j=1:num_dims
  test_point_dir=test_point(j);
  train_point_dir=train_point(j);
  kernel_value=CalculateKernelValue(test_point_dir,train_point_dir,vec_bandwidths(j));
  prod=prod*kernel_value;
end

function [val]=CalculateKernelValue(x,y,bw)

arg=(x-y)^2/(2*bw^2);
val=exp(-arg)/(bw*sqrt(2*pi));

