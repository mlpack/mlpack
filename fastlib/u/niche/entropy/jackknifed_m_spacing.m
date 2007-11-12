% jackknifed_m_spacing() - get an unbiased estimate of entropy
% using jackknifed m-spacing
function H = jackknifed_m_spacing(n);

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

% consider leaving out i

