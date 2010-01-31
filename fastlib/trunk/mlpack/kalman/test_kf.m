function main
% This m-file serves to 
% 1. generate data for main.cc
% but more importantly to
% compare and plot results
% between a fastlib and matlab
% implementation
%
% Follow the following step by step
clc;clear; close all;  

%%%%%%%%%%% 1. Stipulate system params and save as *.csv files%%%%%%%%%%%

%% Duration of demo 
% recall that fastlib
% starts from index 0;
% remember to save transpose of matrices
% because of the way fastlib reads in data
t_tot = 1000; 
csvwrite('t_in',t_tot');      

%% System matrices  
a_mat = diag([0.9, 0.9]);
b_mat = [1;1];
c_mat = [1 1];
q_mat = [8.8125 69.125; 69.125 683.25];
r_mat = 691.06;
s_mat = [30;54.75];

csvwrite('a_in', a_mat'); 
csvwrite('b_in', b_mat');
csvwrite('c_in', c_mat');
csvwrite('q_in', q_mat');
csvwrite('r_in', r_mat'); 
csvwrite('s_in', s_mat');   
 
%% Initial predictions for kf
x_pred_0 = ones(size(a_mat,1),1);
p_pred_0 = diag(repmat(1e-3,size(a_mat,1),1));
y_pred_0 = c_mat*x_pred_0;   
inno_cov_0 = c_mat*p_pred_0*c_mat' + r_mat;

csvwrite('x_pred_0_in', x_pred_0'); 
csvwrite('p_pred_0_in', p_pred_0');
csvwrite('y_pred_0_in', y_pred_0'); 
csvwrite('inno_cov_0_in', inno_cov_0');

%%%%%%%%%%Manually run ./main and load results from directory%%%%%%%%%%
keyboard %type dbcont at command prompt to continue
 
%% These will be used for comparison purposes 
load w_out;
load v_out;
load u_out;
load x_out;
load y_out;
load x_pred_out;
load p_pred_end_out;
load x_hat_out;
load p_hat_end_out;
load y_pred_out;
load inno_cov_end_out;
load k_gain_end_out;

%%%%%%%%%%%%%%%%%%%%%% Run Matlab Equivalent %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% flag = 1: numerically robust implementation
%% flag = 2: numerically non-robust implementation
flag = 1;

% remember to pass in t_tot + 1 and not t_tot
if flag == 1
  [x_ml, y_ml, x_pred_ml, p_pred_ml, x_hat_ml, p_hat_ml, y_pred_ml, inno_cov_ml, k_gain_ml] ...
    = test_kf_sqrt(t_tot +1, a_mat, b_mat, c_mat, q_mat, r_mat, s_mat, x_pred_0, p_pred_0, y_pred_0, inno_cov_0, w_out, v_out, u_out);   
elseif flag == 2
  [x_ml, y_ml, x_pred_ml, p_pred_ml, x_hat_ml, p_hat_ml, y_pred_ml, inno_cov_ml, k_gain_ml] ...
    = test_kf_nosqrt(t_tot +1, a_mat, b_mat, c_mat, q_mat, r_mat, s_mat, x_pred_0, p_pred_0, y_pred_0, inno_cov_0, w_out, v_out, u_out);   
end
   

%%%%%%%%%%%%%%%%%%%%%%%%Visual and Numerical Comparisons%%%%%%%%%%%%%%%%%
%% Trivial checks (just to be sure)
display('......Trivial checks.......')
display('||x_fastlib - x_ml||_2')
display(num2str(norm(x_out - x_ml)))

display('||y_fastlib - y_ml||_2')
display(num2str(norm(y_out - y_ml)))

%% Non-trivial checks 
display('......Non-trivial checks......')
display('||x_pred_fastlib - x_pred_ml||_2')
display(num2str(norm(x_pred_out - x_pred_ml)))
 
display('||p_pred_fastlib - p_pred_ml||_2')
display(num2str(norm(p_pred_end_out - p_pred_ml(:, :, end))))

display('||x_hat_fastlib - x_hat_ml||_2')
display(num2str(norm(x_hat_out - x_hat_ml)))

display('||x_hat_fastlib - x_hat_ml||_2')
display(num2str(norm(p_hat_end_out - p_hat_ml(:, :, end))))

display('||y_pred_fastlib - y_pred_ml||_2')
display(num2str(norm(y_pred_out - y_pred_ml)))

display('||y_pred_fastlib - y_pred_ml||_2')
display(num2str(norm(inno_cov_end_out - inno_cov_ml(:, :, end))))

display('||k_gain_fastlib - k_gain_ml||_2')
display(num2str(norm(k_gain_end_out - k_gain_ml(:, :, end))))

keyboard  
end  % main
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x, y, x_pred, p_pred, x_hat, p_hat, y_pred, inno_cov, k_gain] = ...
    test_kf_sqrt(t_tot, a_mat, b_mat, c_mat, q_mat, r_mat, s_mat, x_pred_0, p_pred_0, y_pred_0, inno_cov_0, w, v, u);
 
%% Define vars
[ny, nx] = size(c_mat);
[nx, nu] = size(b_mat);
x = zeros(nx, t_tot); 
y = zeros(ny, t_tot);
x_pred = zeros(nx, t_tot); x_pred(:, 1) = x_pred_0;
x_hat = zeros(nx, t_tot);
y_pred = zeros(ny, t_tot); y_pred(:, 1) = y_pred_0;
p_pred = zeros(nx, nx, t_tot); p_pred(:, :, 1) = p_pred_0;
p_hat = zeros(nx, nx, t_tot);
inno_cov = zeros(ny, ny, t_tot); inno_cov(:, :, 1) =  inno_cov_0;
k_gain   = zeros(nx, ny, t_tot);

a_eff_mat = a_mat - s_mat*inv(r_mat)*c_mat;
q_eff_mat = q_mat - s_mat*inv(r_mat)*s_mat';
e_mat     = s_mat*inv(r_mat);

  for t = 1:t_tot
    %% Signal generation. As a demo, the system will also 
    %% be an lds with the same params. as the k.f.
    %% althought this doesn't need to be necessarily true.

    % Generate noise signals (trivial)
    w(:, t) = w(:, t); 
    v(:, t) = v(:, t);

    %% y_t = c_matx_t + v_t
    y(:, t) = c_mat*x(:, t) + v(:, t);

    %% Perform measurement update
    pre_mat = zeros(ny+nx, ny+nx);   
    pre_mat(1:ny, 1:ny) = chol(r_mat);
    pre_mat(ny+1:end, 1:ny) = chol(p_pred(:, :, t))*c_mat';
    pre_mat(ny+1:end, ny+1:end) = chol(p_pred(:, :, t));
  
    [dummy, post_mat] = qr(pre_mat);
    post_mat = post_mat.';
    p_hat(:, :, t)  = post_mat(ny+1:end, ny+1:end)*post_mat(ny+1:end, ny+1:end)';
    k_gain(:, :, t) = post_mat(ny+1:end, 1:ny)*inv(post_mat(1:ny, 1:ny));
  
    x_hat(:, t) = x_pred(:, t) + k_gain(:, :, t)*(y(:, t) - y_pred(:, t) );    

    %% Get input (trivial)
    u(:, t) = u(:, t);

    %% Perform time update only if not last time step    
    if t < t_tot
      pre_mat = zeros(2*nx, nx);
      pre_mat(1:nx, 1:nx) = (a_eff_mat*chol(p_hat(:, :, t))')';
      pre_mat(nx+1:end, 1:nx)  = (chol(q_eff_mat)')'; 
      [dummy,post_mat] = qr(pre_mat);
      post_mat = post_mat';
      p_pred(:, :, t+1) = post_mat(1:nx, 1:nx)*post_mat(1:nx, 1:nx)';
      inno_cov(:, :, t+1) = c_mat*p_pred(:, :, t+1)*c_mat' + r_mat;
  
      x_pred(:, t+1) = a_eff_mat*x_hat(:, t) + e_mat*y(:, t) + b_mat*u(:, t);
      y_pred(:, t+1) = c_mat*x_pred(:, t+1);
    end

    %% propagate system one-step ahead only if not last time step    
    if t<t_tot
      x(:, t+1) = a_mat*x(:, t) + b_mat*u(:, t) + w(:, t);
    end
  end
  
end  % test_kf_sqrt

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x, y, x_pred, p_pred, x_hat, p_hat, y_pred, inno_cov, k_gain] = ...
    test_kf_nosqrt(t_tot, a_mat, b_mat, c_mat, q_mat, r_mat, s_mat, x_pred_0, p_pred_0, y_pred_0, inno_cov_0, w, v, u);
 
%% Define vars
[ny, nx] = size(c_mat);
[nx, nu] = size(b_mat);
x = zeros(nx, t_tot); 
y = zeros(ny, t_tot);
x_pred = zeros(nx, t_tot); x_pred(:, 1) = x_pred_0;
x_hat = zeros(nx, t_tot);
y_pred = zeros(ny, t_tot); y_pred(:, 1) = y_pred_0;
p_pred = zeros(nx, nx, t_tot); p_pred(:, :, 1) = p_pred_0;
p_hat = zeros(nx, nx, t_tot);
inno_cov = zeros(ny, ny, t_tot); inno_cov(:, :, 1) =  inno_cov_0;
k_gain   = zeros(nx, ny, t_tot);

a_eff_mat = a_mat - s_mat*inv(r_mat)*c_mat;
q_eff_mat = q_mat - s_mat*inv(r_mat)*s_mat';
e_mat     = s_mat*inv(r_mat);

  for t = 1:t_tot
    %% Signal generation. As a demo, the system will also 
    %% be an lds with the same params. as the k.f.
    %% althought this doesn't need to be necessarily true.

    % Generate noise signals (trivial)
    w(:, t) = w(:, t); 
    v(:, t) = v(:, t);

    %% y_t = c_matx_t + v_t
    y(:, t) = c_mat*x(:, t) + v(:, t);

    %% Perform measurement update
    k_gain(:, :, t) = p_pred(:, :, t)*c_mat'*inv(inno_cov(:, :, t));
    p_hat(:, :, t)  = (eye(nx, nx) - k_gain(:, :, t)*c_mat)*p_pred(:, :, t);  
    x_hat(:, t) = x_pred(:, t) + k_gain(:, :, t)*(y(:, t) - y_pred(:, t) );    

    %% Get input (trivial)
    u(:, t) = u(:, t);

    %% Perform time update only if not last time step    
    if t < t_tot
      p_pred(:, :, t+1) = a_eff_mat*p_hat(:, :, t)*a_eff_mat' + q_eff_mat;
      inno_cov(:, :, t+1) = c_mat*p_pred(:, :, t+1)*c_mat' + r_mat;
  
      x_pred(:, t+1) = a_eff_mat*x_hat(:, t) + e_mat*y(:, t) + b_mat*u(:, t);
      y_pred(:, t+1) = c_mat*x_pred(:, t+1);
    end

    %% propagate system one-step ahead only if not last time step    
    if t<t_tot
      x(:, t+1) = a_mat*x(:, t) + b_mat*u(:, t) + w(:, t);
    end
  end
  
end  % test_kf_nosqrt
 




