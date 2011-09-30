function [] = plotresults_diff(x)

x = [x(:,1) (x(:,3) - x(:,2))];

k_set = unique(x(:,1));

for i = 1:length(k_set)
  k = k_set(i);
  means(i) = mean(x(x(:,1) == k, 2));
  stds(i) = std(x(x(:,1) == k, 2));
end

clf;
errorbar(k_set, means, stds, stds, 'k', 'LineWidth', 1)
hold on;
plot(k_set, means, 'b', 'LineWidth', 3)
