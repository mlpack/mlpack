% gen_data() - generate N Gaussian data points and save to file
function [] = gen_data(N, filename);

mu = 0;
sigma = 1;

data = normrnd(mu * ones(N,1),sigma);
csvwrite(filename, data);
