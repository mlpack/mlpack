function[val]=eval_gaussian_kernel(bw,distance)
dist_by_bw=-(distance/bw)^2;
val=exp(dist_by_bw);