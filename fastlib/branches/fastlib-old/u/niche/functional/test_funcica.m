
% initialize random number generator
rand('state', sum(100*clock))


% sample from laplacian

D = 2;%4;
N = 10000;%1000;

mu = 0;
sigma = 1;
b = sqrt(sigma/2); % Laplacian distribution



% laplacian
clear x px;
for i = 1:D%(D-2)
  x(i,:) = laplacinv(rand(N, 1), mu, b);  % laplacian
 
  %x(i,:) = rand(N, 1);  % uniform random
  
  %x(i,:) = norminv(rand(N, 1), mu, sigma);  %gaussian
end

%x(D-1,:) = norminv(rand(N, 1), mu, sigma);
%x(D,:) = norminv(rand(N, 1), mu, sigma);

% center the sampling distribution
x = x - repmat(mean(x')', 1, N);



% generate b-spline basis curves
t = linspace(0,1,1000);
%t = linspace(0, 1, 200);


load s1s2_10;
s1 = @(x) sqrt(2) * sin(pi * 5 * x);
s2 = @(x) sqrt(2) * cos(pi * 5 * x);
%s = [s1(t); s2(t)]';
%s3 = @(x) sqrt(2) * sin(pi * 40 * x);
%s = [s1(t); s2(t); s3(t)]';

%s3 = @(x) sqrt(2) * sin(pi * 40 * x);
%s4 = @(x) sqrt(2) * cos(pi * 40 * x);
s = [s1(t); s2(t)]';%; s3(t); s4(t)]';


%z = normrnd(zeros(length(t), N), .2);

data = s * x;


%random_walks = zeros(length(t), N);
%for i = 1:N
%  if mod(i, 10) == 0
%    disp(i);
%  end
%  for j = 2:length(t)
%    random_walks(j,i) = random_walks(j - 1, i) + normrnd(0, .1);
%  end
%end

%data = data + random_walks;


%noise = 1 * normrnd(zeros(length(t),N), 1);
%data = data + noise;
%return;
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
%p = round(0.9 * length(t)); % hardcoded for now
%p = round(0.95 * length(t));
p = 120;

mybasis = create_bspline_basis([min(t) max(t)], p, 4);
basis_curves = eval_basis(t, mybasis);
basis_inner_products = full(eval_penalty(mybasis, int2Lfd(0)));

myfd_data = data2fd(data, t, mybasis);


lambda = 1e-6;%1e-9;%0;%5.9609e-10;
myfdPar = fdPar(mybasis, 2, lambda);

[ic_curves, ic_coef, ic_scores, ...
 pc_coef, pc_curves, pc_scores, W] = ...
    funcica(t, myfd_data, 4, basis_curves, myfdPar, ...
	    basis_inner_products);

%p_small = size(ic_coef, 2);

rescale_ics;

ic1_s1_prod = l2_fnorm(t, ic_curves(:,1), s1(t));
ic1_s2_prod = l2_fnorm(t, ic_curves(:,1), s2(t));
if abs(ic1_s1_prod) > abs(ic1_s2_prod)
  ic1_sign = sign(ic1_s1_prod);
  ic2_sign = sign(l2_fnorm(t, ic_curves(:,2), s2(t)));
else
  ic1_sign = sign(ic1_s2_prod);
  ic2_sign = sign(l2_fnorm(t, ic_curves(:,2), s1(t)));
end

pc1_s1_prod = l2_fnorm(t, pc_curves(:,1), s1(t));
pc1_s2_prod = l2_fnorm(t, pc_curves(:,1), s2(t));
if abs(pc1_s1_prod) > abs(pc1_s2_prod)
  pc1_sign = sign(pc1_s1_prod);
  pc2_sign = sign(l2_fnorm(t, pc_curves(:,2), s2(t)));
else
  pc1_sign = sign(pc1_s2_prod);
  pc2_sign = sign(l2_fnorm(t, pc_curves(:,2), s1(t)));
end


figure(1);
clf;
hold on;
plot(t, s1(t), 'r');
plot(t, s2(t), 'g');
plot(t, ic1_sign * ic_curves(:,1), 'k--');
plot(t, ic2_sign * ic_curves(:,2), 'k--');
plot(t, pc1_sign * pc_curves(:,1), 'b');
plot(t, pc2_sign * pc_curves(:,2), 'm');

%legend('S_1(t)', 'S_2(t)', 'IC_1(t)', 'IC_2(t)', 'PC_1(t)', 'PC_2(t)');





% rescale the utilized parts of pc_coef such that
% the pc_curves square integrate to 1

%for j = 1:p_small
%  pc_coef_j = pc_coef(:,j)';
%  alpha = sqrt(sum(sum((pc_coef_j' * pc_coef_j) .* basis_inner_products)));
%  pc_coef(j,:) = pc_coef(j,:) / alpha;
%end


