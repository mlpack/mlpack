function[]=process_packet_data()
%This file  processes the data packet_data file. it takes the logarithms of
%the distances (the first 22 features

data=csvread('aggr_all_features_scaled.data');
[num_points,num_features]=size(data);
display(num_points);
%The last feature is the label. We shall take the logarithm of the first 21
%features
data_new=zeros(1,num_features);
for i=1:num_points
   
    temp=log(data(i:i,1:num_features-1));
    temp=[temp,0]; %adding 1 zeros
    %The last feature is the label. Since the dataset has been labeled
    %+1,-1. We shall convert it to +1,0
    if(data(i:i,num_features)==1)
        temp(end)=1;
    else
        temp(end)=0;
    end
    
    if(i==1)
        data_new=temp;
    else
        [data_new]=[data_new;temp];
    end
end
csvwrite('aggr_all_features_scaled.data',data_new);