function [clusters_stratified, clusters, cluster_rank] = find_emst_clusters(emst_mat, k)

% Given the output of the fastlib EMST computation and a number of clusters
% k, returns a vector of cluster labels 1:k
% 
% IMPORTANT: Assumes the points are indexed 0 to N-1
%
% INPUTS:
%% emst_mat - the matrix output by the fastlib emst algorithm
%% k - number of desired clusters
%
% OUTPUTS:
%% clusters - a vector of N cluster labels, where N is the number of points
%% in the MST


[num_edges, blah] = size(emst_mat);
num_points = num_edges + 1;

% find largest edges

% first column = index, second = length
largest = zeros(k-1,2);
smallest_ind = 1;
smallest_val = 0;
for i=1:num_edges
    
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
split_emst(largest(:,1),:) = [];
% Move points so they are indexed by one instead of zero
split_emst(:,1:2) = split_emst(:,1:2) + ones(num_edges-k+1,2);

clusters = 1:num_points;
clusters = clusters';
cluster_rank = zeros(num_points,1);

num_clusters_ideal = num_points;

for i=1:num_edges-k+1

%i
%    clusters(split_emst(i,1))
%cluster_rank(split_emst(i,1))
%    clusters(split_emst(i,2))
%cluster_rank(split_emst(i,2))

    [clusters, cluster_rank] = emst_union(clusters, cluster_rank, split_emst(i,1), split_emst(i,2));
    num_clusters_ideal = num_clusters_ideal - 1;

%clusters(split_emst(i,1))
%cluster_rank(split_emst(i,1))
%clusters(split_emst(i,2)) 
%cluster_rank(split_emst(i,2))   

end

% flatten out the list
for i = 1:num_points

    [par, clusters] = emst_find(i, clusters);

end

clusters_stratified = {};
cluster_label_encountered = [];
for i = 1:num_points
    if length(find(cluster_label_encountered == clusters(i))) == 0
        list_labels = find(clusters == clusters(i));
        clusters_stratified{end + 1} = list_labels;
        cluster_label_encountered = [cluster_label_encountered clusters(i)];
    end;
end;