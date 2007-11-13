% jackknifed_m_spacing() - get an unbiased estimate of entropy
% using jackknifed m-spacing
function H = jackknifed_m_spacing(n, j);

randn('seed', sum(100*clock));
X = normrnd(zeros(n, 1), 1);
Z = sort(X, 'ascend');
m = round(sqrt(n)/2);

% the vanilla m-spacing estimator for reference
%{
sum_logs = 0;
for i = 1:n
  if (i + m) > n
    sum_logs = sum_logs + log( (n/(2*m)) * (Z(n) - Z(i-m)));
  elseif (i-m) < 1
    sum_logs = sum_logs + log( (n/(2*m)) * (Z(i+m) - Z(1)));
  else    
    sum_logs = sum_logs + log( (n/(2*m)) * (Z(i+m) - Z(i-m)));
  end
end
%}

H = 1;

% consider leaving out Z(j)

sum_logs = 0;

%j = 10;

% business as usual for i <= j-m-1 and i >= j+m+1
for i=1:(j-m-1)
  if (i + m) > n
    if j == n
      sum_logs = sum_logs + log( ((n-1)/(2*m)) * (Z(n-1) - Z(i-m)));
    else
      sum_logs = sum_logs + log( ((n-1)/(2*m)) * (Z(n) - Z(i-m)));
    end
  elseif (i-m) < 1
    if j == 1
      sum_logs = sum_logs + log( ((n-1)/(2*m)) * (Z(i+m) - Z(2)));
    else
      sum_logs = sum_logs + log( ((n-1)/(2*m)) * (Z(i+m) - Z(1)));
    end
  else    
    sum_logs = sum_logs + log( ((n-1)/(2*m)) * (Z(i+m) - Z(i-m)));
  end
end

for i = max(1,(j-m)):min(n-1,(j+m-1))
  if (i+m+1) > n
    if j == n      
      sum_logs = sum_logs + log( ((n-1)/(2*m)) * (Z(n-1) - Z(i-m)));
    else
      sum_logs = sum_logs + log( ((n-1)/(2*m)) * (Z(n) - Z(i-m)));
    end
  elseif (i-m) < 1
    if j == 1
      sum_logs = sum_logs + log( ((n-1)/(2*m)) * (Z(i+m+1) - Z(2)));
    else
      sum_logs = sum_logs + log( ((n-1)/(2*m)) * (Z(i+m+1) - Z(1)));
    end
  else
    sum_logs = sum_logs + log( ((n-1)/(2*m)) * (Z(i+m+1) - Z(i-m)));
  end  
end

for i=(j+m+1):n
  if (i + m) > n
    if j == n
      sum_logs = sum_logs + log( ((n-1)/(2*m)) * (Z(n-1) - Z(i-m)));
    else
      sum_logs = sum_logs + log( ((n-1)/(2*m)) * (Z(n) - Z(i-m)));
    end
  elseif (i-m) < 1
    if j == 1
      sum_logs = sum_logs + log( ((n-1)/(2*m)) * (Z(i+m) - Z(2)));
    else
      sum_logs = sum_logs + log( ((n-1)/(2*m)) * (Z(i+m) - Z(1)));
    end
  else    
    sum_logs = sum_logs + log( ((n-1)/(2*m)) * (Z(i+m) - Z(i-m)));
  end
end

  
H = sum_logs / (n-1);



Z = Z([1:(j-1) (j+1):n]);

n = n-1;

sum_logs = 0;

for i = 1:n
  if (i + m) > n
    sum_logs = sum_logs + log( (n/(2*m)) * (Z(n) - Z(i-m)));
  elseif (i-m) < 1
    sum_logs = sum_logs + log( (n/(2*m)) * (Z(i+m) - Z(1)));
  else    
    sum_logs = sum_logs + log( (n/(2*m)) * (Z(i+m) - Z(i-m)));
  end
end

H_correct = sum_logs/n;

H = H - H_correct;