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
z = normrnd(zeros(length(t), N), .2);

data = s * x;

data = data + z;

indices = 1:10:1000;
t = t(indices);
data = data(indices,:);



%data = noisy_data;
