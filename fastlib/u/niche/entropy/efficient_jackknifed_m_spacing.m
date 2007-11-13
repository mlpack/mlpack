% jackknifed_m_spacing() - efficiently get an unbiased estimate of entropy
% using jackknifed m-spacing

function H = efficient_jackknifed_m_spacing(n);

randn('seed', sum(100));%*clock));
X = normrnd(zeros(n, 1), 1);
Z = sort(X, 'ascend');
m = round(sqrt(n)/2);



H = zeros(1,n);
H_correct = zeros(1,n);


sum_logs_before = 0;

sum_logs_after = 0;

logs_after = zeros(n - m - 1, 1);
for i = (m+2):n
  if (i + m) > n
    logs_after(i-1-m) = log(Z(n) - Z(i-m));
  else    
    logs_after(i-1-m) = log(Z(i+m) - Z(i-m));
  end
end

sum_logs_after = sum(logs_after);



for j=1:n

   
  % business as usual for i <= j-m-1 and i >= j+m+1
  if (j-m-1) >= 1
    i = j-m-1;
    if (i-m) < 1
      sum_logs_before = sum_logs_before + log(Z(i+m) - Z(1));
    else    
      sum_logs_before = sum_logs_before + log(Z(i+m) - Z(i-m));
    end
  end

  % this part must always be recalculated
  sum_logs_at = 0;
  for i = max(1,(j-m)):min(n-1,(j+m-1))
    if (i+m+1) > n
      if j == n      
	sum_logs_at = sum_logs_at + log(Z(n-1) - Z(i-m));
      else
	sum_logs_at = sum_logs_at + log(Z(n) - Z(i-m));
      end
    elseif (i-m) < 1
      if j == 1
	sum_logs_at = sum_logs_at + log(Z(i+m+1) - Z(2));
      else
	sum_logs_at = sum_logs_at + log(Z(i+m+1) - Z(1));
      end
    else
      sum_logs_at = sum_logs_at + log(Z(i+m+1) - Z(i-m));
    end  
  end

  % subtract the last added item
  if (j >= 2) && (j <= n - m)
    sum_logs_after = sum_logs_after - logs_after(j - 1);
  end

  
  H(j) = ((sum_logs_before + sum_logs_at + sum_logs_after)/(n-1)) + ...
	 log((n-1)/(2*m));
  
  sum_logs_before_array(j) = sum_logs_before;
  sum_logs_at_array(j) = sum_logs_at;
  sum_logs_after_array(j) = sum_logs_after;



end




%{
n_minus_1 = n-1;

for j=1:n
  
  Z_minus_j = Z([1:(j-1) (j+1):n]);
  
  sum_logs = 0;
  
  for i = 1:n_minus_1
    if (i + m) > n_minus_1
      sum_logs = sum_logs + log(Z_minus_j(n_minus_1) - Z_minus_j(i-m));
    elseif (i-m) < 1
      sum_logs = sum_logs + log(Z_minus_j(i+m) - Z_minus_j(1));
    else    
      sum_logs = sum_logs + log(Z_minus_j(i+m) - Z_minus_j(i-m));
    end
  end
  
  H_correct(j) = (sum_logs/n_minus_1) + log(n_minus_1/(2*m));
  
  error(j) = H(j) - H_correct(j);
  
end
%}
