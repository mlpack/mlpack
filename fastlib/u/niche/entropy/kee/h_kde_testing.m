% h_kde_testing() - produce many entropy estimates using the
% entropy of a kernel density estimate of the true distribution and
% different sampling distributions for the expectation of
% -log(f(x))

function [p_values, h_stats] = h_kde_testing(N,h);

x = normrnd(zeros(N,1), 1);


% to speed up computations for ensemble testing, first store the
% densities at our sample points

f_x = zeros(N,1);

for i=1:N
  sum = 0;
  for j=1:N
    sum = sum + exp(-((x(i) - x(j))^2) / (2*(h^2)));
  end
  f_x(i) = sum;
end

f_x = f_x / (N * h * sqrt(2 * pi));

disp('done generating f_x');

t1 = clock;

p_values = linspace(0.01, 1, 100);
h_stats = zeros(100,10);

for p_index = 1:length(p_values)
  K = ceil(p_values(p_index) * N);

  for j=1:10
    indices = randint(K, 1, N) + 1;
    
    sum_logs = 0;
    for i=1:K
      sum_logs = sum_logs + log(f_x(indices(i)));
    end
  
    h_stats(p_index, j) = - sum_logs / K;
  end
end

t2 = clock;

disp(t2 - t1);
