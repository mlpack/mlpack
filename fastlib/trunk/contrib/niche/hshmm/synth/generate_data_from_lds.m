function [q x] = ...
    generate_data_from_lds(A, C, Q, R, ...
			   mu_0, Sigma_0, ...
			   T, mu_w, mu_v)
%function [q x] = ...
%    generate_data_from_lds(A, C, Q, R, ...
%			   mu_0, Sigma_0, ...
%			   T, mu_w, mu_v)
%
% T defaults to 1e4
% mu_w defaults to a zero vector
% mu_v defaults to a zero vector
% Note that for consistency with Ghahramani's LDS code, it is
% necessary to avoid passing non-zero mu_w and mu_v


n_dims_latent = size(C, 2);
n_dims_obs = size(C, 1);


if ~exist('T')
  T = 1e4;
  fprintf(['parameter ''T'' not passed, setting it to ' ...
	   '%d.\n'], T);
end

if ~exist('mu_w')
  mu_w = zeros(1, n_dims_latent);
elseif (size(mu_w, 2) == 1)
    mu_w = mu_w';
end

if ~exist('mu_v')
  mu_v = zeros(1, n_dims_obs);
elseif (size(mu_v, 2) == 1)
    mu_v = mu_v';
end


if ~isequal(size(A), [n_dims_latent, n_dims_latent]) || ...
      ~isequal(size(Q), [n_dims_latent, n_dims_latent]) || ...
      ~isequal(size(R), [n_dims_obs, n_dims_obs]) || ...
      ~isequal(size(mu_0), [n_dims_latent 1]) || ...
      ~isequal(size(Sigma_0), [n_dims_latent n_dims_latent]) || ...
      ~isequal(size(mu_w), [1 n_dims_latent]) || ...
      ~isequal(size(mu_v), [1 n_dims_obs])
  fprintf(['Error: czech the dimensions! parameters are ' ...
	   'inconsistently sized!\n']);
  q = [];
  x = [];
  return;
end


q = zeros(n_dims_latent, T);
x = zeros(n_dims_obs, T);


q(:, 1) = mvnrnd(mu_0, Sigma_0);
x(:, 1) = C * q(:, 1) + mvnrnd(mu_v, R);
for t = 2:T
  q(:, t) = A * q(:, t-1) + mvnrnd(mu_w, Q)';
  x(:, t) = C * q(:, t) + mvnrnd(mu_v, R)';
end

q = q';
x = x';


%mu_0 = mu_0';
%save mu_0.dat mu_0 -ascii;

%Sigma_0 = Sigma_0';
%save Sigma_0.dat Sigma_0 -ascii;

%A = A';
%save A.dat A -ascii;

%C = C';
%save C.dat C -ascii;

%Q = Q';
%save Q.dat Q -ascii;

%R = R';
%save R.dat R -ascii;

%q = q';
%save Q.dat q -ascii;

%x = x';
%save X.dat x -ascii;

