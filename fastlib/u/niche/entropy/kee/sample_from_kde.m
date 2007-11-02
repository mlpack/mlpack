% using the kernel density estimator, sample densities from the
%distribution induced by the estimator
% compare this to the gaussian kernel convolution


K = 1;

differences = zeros(1,K);
% N     mean   std
% 10    .0451  .0242
% 100   .0101  .0042
% 1000  .0055  .0013
% 1500  .0053  8.1601e-04
% 2000  .0050  8.7302e-04

for k = 1:K
  
  N = 10000;
'start finding h'  
  h = find_h_D(X);
  'found h'
%   X = normrnd(zeros(N,1),1);
  
  p_X = kde(X, h);
  
  M = 1 * N;
  
  sum_s = 0;
  sum_p_s = 0;
  
  s_array   = zeros(1, M);
  p_s_array = zeros(1, M);
  
  % query_points = X;
  query_points = linspace(-5,5,M);
  
  for i = 1 : M
    select = randint(1, 1, [1 N]);
    
    % s = norminv(rand, X(select), h);
    s = query_points(i);
    s_array(i) = s;
    
    %  query the kde at point s
    
    p_s = 0;
    for j = 1:N
      p_s = p_s + exp(-((s - X(j))^2) / (2*(h^2)));
    end
    p_s = p_s / (N*h*sqrt(2*pi));
    
    sum_s = sum_s + s;
    sum_p_s = sum_p_s + p_s;
    p_s_array(i) = p_s;
    
  end
  
  disp('E_f_hat[x]');
  disp(sum_s / M);
  disp('E_f_hat[f_hat(x)]');
  disp(sum_p_s / M);
  
  
  
end

K2_X = kde(X, 2*h) * (N) * (2*h);

  
  
star_kernel_sum = sum(K2_X);
  
star_kernel_sum = star_kernel_sum / (2 * h * (N * (N)))
  
differences(k) = (sum_p_s / M) - star_kernel_sum;
  
disp('E_f_hat[f_hat(x)] - star_kernel_sum');
disp(differences(k));
  
  

% explicit star_kernel_sum with maximal sampling

p_X = zeros(size(X));

i = 1;
for i = 1:M
  q = query_points(i);
  collect = 0;
  for j = 1:N
    collect = collect + exp(-((q - X(j))^2) / (2*((2*h)^2)));
  end
  p_X(i) = collect;
end

K2_X_2 = p_X / (N * (2*h) * sqrt(2*pi));


disp('K2_X_2');
disp(sum(K2_X_2) / M);




%disp('mean(E_f_hat[f_hat(x)] - star_kernel_sum)');
%disp(mean(differences));

%disp('std(E_f_hat[f_hat(x)] - star_kernel_sum)');
%disp(std(differences));

finite_difference = diff(ppval(fnint(spline(s_array, p_s_array .* ...
					    p_s_array)), [min(s_array), ...
		    max(s_array)]));


disp('finite difference');
disp(finite_difference);

finite_difference_difference = finite_difference - star_kernel_sum;
disp('finite_difference - star_kernel_sum');
disp(finite_difference_difference);
