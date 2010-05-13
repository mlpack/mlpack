function [A, B, C] = generate_data(m)

A_mu = 1;
A_sigma = 1;

alpha = 0.5;
B_sigma = 0.5;

beta = 0.9;
C_sigma = 1;

A = normrnd(A_mu, A_sigma, m, 1);
B = alpha * A + normrnd(0, B_sigma, m, 1);
C = beta * B + normrnd(0, C_sigma, m, 1);
