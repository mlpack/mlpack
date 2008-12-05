function [clusters] = find_emst_clusters(emst_mat, k)

% Given the output of the fastlib EMST computation and a number of clusters
% k, returns a vector of cluster labels 1:k
% 
% INPUTS:
%% emst_mat - the matrix output by the fastlib emst algorithm
%% k - number of desired clusters
%
% OUTPUTS:
%% clusters - a vector of N cluster labels, where N is the number of points
%% in the MST


[num_points, blah] = size(emst_mat);

%clusters = zeros(num_points-k+1,1);

% find largest edges

% first column = index, second = length
largest = zeros(k-1,2);
smallest_ind = 1;
smallest_val = 0;
for i=1:num_points
    
    if (emst_mat(i,3) > smallest_val)
        
        largest(smallest_ind,1) = i;
        largest(smallest_ind,2) = emst_mat(i,3);
                
        new_smallest_ind = 1;
        new_smallest_val = largest(1,2);
        
        for j=2:k-1
            
            if (largest(j,2) < new_smallest_val)
               
                new_smallest_val = largest(j,2);
                new_smallest_ind = j;
                
            end
            
        end
        
        smallest_ind = new_smallest_ind;
        smallest_val = new_smallest_val;
        
    end
    
end


split_emst = emst_mat;
split_emst(largest(:,1),:) = []


clusters = 1:num_points+1;
clusters = clusters'

cluster_rank = zeros(num_points+1,1);

sprintf('calling union');
for i=1:num_points-k+1
    
    [clusters, cluster_rank] = emst_union(clusters, cluster_rank, split_emst(i,1), split_emst(i,2));
    
end

