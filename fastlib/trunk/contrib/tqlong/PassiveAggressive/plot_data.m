function plot_data(samples, labels) 
samples = samples(:, 1:2); % only plot in 2 dimensions
plot(samples(labels>0), '+b', 'LineWidth', 2, 'MarkerSize', 12); hold on;
plot(samples(labels<0), 'dr', 'LineWidth', 2, 'MarkerSize', 12); hold off;
end