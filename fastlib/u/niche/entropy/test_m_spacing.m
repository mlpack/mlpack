function mean_H_minus_opt = test_m_spacing(num_trials, n);

opt=log(sqrt(2*pi*exp(1)));

h = zeros(num_trials,1);

for i=1:num_trials
  h(i) = m_spacing(n);
end

mean_H_minus_opt = mean(h) - opt;
