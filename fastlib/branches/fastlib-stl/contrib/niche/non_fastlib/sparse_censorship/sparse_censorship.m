function [] = sparse_censorship(X, K)
%function [] = sparse_censorship(X)
%
% X - word counts, stored as sparse (# docs) x (# vocab words)
%     since X is sparse, storing the transpose instead may not matter


% We use Empirical Bayes to treat theta, beta, and eta.
% Each of these 3 parameters is sparse. We use the Laplace prior,
% via a Gaussian prior with Exponentially distributed variance.
% Their respective variance parameters are nu, xi, and tau. The
% variance parameters' respective Exponential distribution
% parameters are sigma, iota, and gamma.
%

% D is the number of documents
% V is the number of vocabulary words
[D, V] = size(X);

% 1) Initialize parameters

% Initialize theta for each document

% Initialize beta for each topic

% Initialize eta for each topic and publisher




% 2) Variational EM Loop

% 2 a) E-Step

% Update variational distributions

% Compute expectations needed for model parameter updates

% 2 b) M-step

% Update model parameters

% 2 c) Go back to E-Step

