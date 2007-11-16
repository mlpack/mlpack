% get_vasicek_entropy_estimate() - estimate entropy using vasicek's
% entropy estimator
function h = get_vasicek_entropy_estimate(X);

n = length(X);
m = round(sqrt(n)/2);
Z = sort(X, 'ascend');

sum_logs = 0;

for i = 1:n
  if (i + m) > n
    sum_logs = sum_logs + log(Z(n) - Z(i-m));
  elseif (i-m) < 1
    sum_logs = sum_logs + log(Z(i+m) - Z(1));
  else    
    sum_logs = sum_logs + log(Z(i+m) - Z(i-m));
  end
end

  

h = (sum_logs/n) + log(n/(2*m));
