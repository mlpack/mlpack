% initialize random number generator
rand('state', sum(100*clock))


% sample from laplacian

D = 2;
N = 10000;

mu = 0;
sigma = 1;
b = sqrt(sigma/2); % recent edit - see the wikipedia page for
                   % Laplacian distribution



% laplacian
clear x px;
for i = 1:D
  x(i,:) = laplacinv(rand(N, 1), mu, b);  % laplacian
 
  %x(i,:) = rand(N, 1);  % uniform random
  
  %x(i,:) = norminv(rand(N, 1), mu, sigma);  %gaussian
end

% center the sampling distribution
x = x - repmat(mean(x')', 1, N);



% generate b-spline basis curves
t = linspace(0,1,1000);


load s1s2_10;
s = [s1(t); s2(t)]';
%z = normrnd(zeros(length(t), N), .2);

data = s * x;

%data = data + z;

%indices = 1:10:1000;
%t = t(indices);
%data = data(indices,:);



%data = noisy_data;











%function [h_Y1_train_set, h_Y2_train_set, h_Y_train_set, h_Y1_test_set, ...
%	  h_Y2_test_set, h_Y_test_set, ...
%	  h_P1_train_set, h_P2_train_set, h_P_train_set, h_P1_test_set, ...
%	  h_P2_test_set, h_P_test_set] = ...
%    test_smooth_funcica(data, t, s, basis_inner_products);

N = size(data, 2);
p = 30; % hardcoded for now

mybasis = create_bspline_basis([min(t) max(t)], p, 4);
basis_curves = eval_basis(t, mybasis);
basis_inner_products = full(eval_penalty(mybasis, int2Lfd(0)));

myfd_data = data2fd(data, t, mybasis);


lambda = 0;
myfdPar = fdPar(mybasis, 2, lambda);
fprintf('calling funcica\n');
[ic_curves, ic_coef, Y, h_Y, ...
 pc_coef, pc_curves, pc_scores, W, whitening_transform] = ...
    funcica(t, s, myfd_data, p, basis_curves, myfdPar, ...
	    basis_inner_products);
fprintf('funcica returned\n');
  
p_small = size(ic_coef, 2);

pc_curves = basis_curves * pc_coef;

ic_curves = pc_curves(:,1:2) * W';

pc_coef = pc_coef';


    
% rescale the utilized parts of pc_coef such that
% the pc_curves square integrate to 1

for j = 1:p_small
  pc_coef_j = pc_coef(j,:);
  alpha = sqrt(sum(sum((pc_coef_j' * pc_coef_j) .* basis_inner_products)));
  pc_coef(j,:) = pc_coef(j,:) / alpha;
end

scores = pc_scores';

Y_scores = W * scores(1:2,:);


magnitude = sum(sum(scores .^ 2));


%{
h_P1 = ...
    get_vasicek_entropy_estimate_std(scores(1,:));
h_P2 = ...
    get_vasicek_entropy_estimate_std(scores(2,:));
h_P_all = h_P1 + h_P2;

h_Y1 = ...
    get_vasicek_entropy_estimate_std(Y_scores(1,:));
h_Y2 = ...
    get_vasicek_entropy_estimate_std(Y_scores(2, :));
h_Y_all = h_Y1 + h_Y2;
%}

