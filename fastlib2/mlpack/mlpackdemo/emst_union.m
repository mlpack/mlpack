function [cluster_out, rank_out] = emst_union(cluster, rank, x, y)

rank_out = rank;
[xroot, temp_cluster, xrank] = emst_find(x, cluster, rank);
[yroot, cluster_out, yrank] = emst_find(y, temp_cluster, rank);

if (xrank > yrank)
    
    cluster_out(y) = xroot;
    
elseif (xrank < yrank)
    
    cluster_out(x) = yroot;
    
elseif (xroot ~= yroot)
    
    cluster_out(y) = xroot;
    rank_out(xroot) = rank_out(xroot) + 1;
    
end