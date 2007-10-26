[p_values, h_stats] = h_kde_testing(1 * 1e4, .5);

hold off;
figure(1);
plot(p_values, var(h_stats'));

figure(2);
plot(p_values, mean(h_stats'));
hold on;
plot(p_values, ...
     repmat(log(sqrt(2*pi*exp(1))), 1, 100), 'r');
