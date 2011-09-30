function [log_k_set, log_means] = plot_loglog(x)

x = [x(:,1) (x(:,3) - x(:,2))];

k_set = unique(x(:,1));

means = zeros(size(k_set));
for i = 1:length(k_set)
  k = k_set(i);
  means(i) = mean(x(x(:,1) == k, 2));
end

clf;
loglog(k_set, means);

log_k_set = log(k_set);
log_means = log(means);
