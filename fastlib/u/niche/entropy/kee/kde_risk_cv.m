% kde_risk_cv() - cross validate to select the optimal bandwidth
% according to risk minimization
function [risk] = kde_risk_cv(N, h_array);

x = normrnd(zeros(N,1), 1);

num_h = length(h_array);

risk = zeros(num_h,1);

for i=1:num_h
%  h_array(i)
  risk(i) = kde_risk(x, h_array(i));
end
