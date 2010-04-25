function [cluster_out, rank_out] = emst_union(cluster, cluster_rank, x, y)

rank_out = cluster_rank;
[xroot, temp_cluster] = emst_find(x, cluster);
[yroot, cluster_out] = emst_find(y, temp_cluster);
xrank = cluster_rank(xroot);
yrank = cluster_rank(yroot);

if (xroot == yroot)

     sprintf('xroot equals yroot');

elseif (xrank > yrank)
    
    cluster_out(yroot) = xroot;
    
elseif (xrank < yrank)
    
    cluster_out(xroot) = yroot;
    
elseif (xroot ~= yroot)
    
    cluster_out(yroot) = xroot;
    rank_out(xroot) = rank_out(xroot) + 1;

end

