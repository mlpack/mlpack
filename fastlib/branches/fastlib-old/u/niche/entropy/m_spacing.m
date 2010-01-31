% m_spacing() - estimate entropy using m-spacing
function H = m_spacing(n);

randn('seed', sum(100*clock));

X = normrnd(zeros(n, 1), 1);

m = round(sqrt(n)/2);



Z = sort(X, 'ascend');

sum_logs = 0;
%for i = (m+1):(n-m)
for i = 1:n
  if (i + m) > n
    sum_logs = sum_logs + log( (n/(2*m)) * (Z(n) - Z(i-m)));
  elseif (i-m) < 1
    sum_logs = sum_logs + log( (n/(2*m)) * (Z(i+m) - Z(1)));
  else    
    sum_logs = sum_logs + log( (n/(2*m)) * (Z(i+m) - Z(i-m)));
  end
end

H = sum_logs / n;
%H = sum_logs / (n-2*m);

% Vasicek's bias correction

sum_psi = 0;
for i=1:m
  sum_psi = sum_psi + psi(i+m-1);
end

bias = ...
    + log(n) ...
    - log(2*m) ...
    + (1 - 2*m/n) * psi(2*m) ...
    - psi(n+1) ...
    + (2/n) * sum_psi;



H = H - bias;