function [means, stds] = plotresults(x)

k_set = unique(x(:,1));

for i = 1:length(k_set)
  k = k_set(i);
  means(i,1:2) = mean(x(x(:,1) == k, 2:3));
  stds(i,1:2) = std(x(x(:,1) == k, 2:3));
end

clf;
errorbar(k_set, means(:,1), stds(:,1), stds(:,1), 'b')
hold on;
errorbar(k_set, means(:,2), stds(:,2), stds(:,2), 'r')
