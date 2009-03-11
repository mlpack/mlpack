function [parent, cluster_out] = emst_find(x, cluster)

cluster_out = cluster;

if (cluster(x) == x)
   
    parent = x;
    
else
    
    [parx, cluster_out] = emst_find(cluster(x), cluster);
    parent = parx;
    cluster_out(x) = parx;
    
end
