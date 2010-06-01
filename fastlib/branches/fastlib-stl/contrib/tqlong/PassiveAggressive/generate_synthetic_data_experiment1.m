function [samples, labels] = generate_synthetic_data_experiment1(N, noise_sigma, filename)

positive_mean = [1 1];
negative_mean = [-1 -1];
covariance = [0.2 0; 0 2];
noise_covariance = noise_sigma*eye(2);

labels = rand(N,1) > 0.5;
positive_sample = mvnrnd(positive_mean, covariance, N);
negative_sample = mvnrnd(negative_mean, covariance, N);
noise = mvnrnd(zeros(1, 2), noise_covariance, N);

samples = zeros(N, 2);
samples(labels,:) = positive_sample(labels,:);
samples(~labels,:) = negative_sample(~labels,:);
samples = samples+noise;
labels = labels*2-1;
if nargin > 2 % there is a filename, write data to the file
    data = [samples labels];
    save(filename, 'data', '-ASCII');
end

end