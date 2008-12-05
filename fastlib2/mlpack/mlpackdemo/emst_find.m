function [parent, cluster_out, rank] = emst_find(x, cluster, rank)

cluster_out = cluster;

if (cluster(x) == x)
   
    parent = x;
    rank = rank(x);
    
else
    
    [parx, cluster_out, rank] = emst_find(cluster(x), cluster, rank);
    parent = parx;
    cluster_out(x) = parx;
    
end