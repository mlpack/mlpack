% nn_entropy_estimator() - estimate entropy using nearest neighbor distance
function H = nn_entropy_estimator(X);

CE = 0.577215665; % Euler's constant

X = X / std(X);

N = length(X);

X = sort(X, 'ascend');

sum_ln_dist = log(X(2) - X(1));

for i=2:N-1
  sum_ln_dist = ...
      sum_ln_dist + log(min([X(i+1) - X(i) , X(i) - X(i-1)]));
end  

sum_ln_dist = sum_ln_dist + log(X(N) - X(N-1));
  
H = (sum_ln_dist / N) + log(2*N) + CE;