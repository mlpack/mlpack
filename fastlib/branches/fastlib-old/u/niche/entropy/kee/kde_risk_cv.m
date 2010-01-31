% kde_risk_cv() - cross validate to select the optimal bandwidth
% according to risk minimization
function [risk] = kde_risk_cv(x, h_array);

%x = load('/home/niche/scaled_refined_astroset_20k.ds');

%x = normrnd(zeros(N,1), 1);

num_h = length(h_array);

risk = zeros(num_h,1);

for i=1:num_h
%  h_array(i)
  risk(i) = kde_risk(h_array(i), x, size(x,2));
end
