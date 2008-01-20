function main
clc;clear; close all; 
flag = 1;
randn('state',100);  
%Step 1: Run This
A = diag([0.9, 0.9]);
B = [1;1];
C = [1 1];
D = 1;
Q = [8.8125 69.125; 69.125 683.25];
R = 691.06;
S = [30;54.75];
  
%A = diag([0.9]);
%B = [21];
%C = [1];
%D = 5;
%Q = [1.25 ];
%R = 26.25;
%S = [0.7];

x_pred_init = ones(size(A,1),1);
P_pred_init = diag(repmat(1e-3,size(A,1),1));

T = 1000;
if flag == 1
  [x_orig,x_pred_orig,x_hat_orig,y_pred_orig,y_orig,P_pred_orig, P_hat_orig, inno_cov_orig,K_orig] = test_kf_sqrt(A,B,C,D,Q,R,S,x_pred_init,P_pred_init,T);
else
  [x_orig,x_pred_orig,x_hat_orig,y_pred_orig,y_orig,P_pred_orig, P_hat_orig, inno_cov_orig,K_orig] = test_kf_no_sqrt(A,B,C,D,Q,R,S,x_pred_init,P_pred_init,T);
end

%Step 2: Compile ./main manually
 
%Step 3: Run the following
keyboard
load x_pred; 
load x_hat; 
load y_pred;
load K_end;
load P_pred_end;
load P_hat_end;
load inno_cov_end; 
 
[norm(x_pred-x_pred_orig) norm(y_pred-y_pred_orig) norm(x_hat-x_hat_orig)]
[norm(inno_cov_orig(:,:,end) - inno_cov_end) norm(P_pred_orig(:,:,end) - P_pred_end) norm(P_hat_orig(:,:,end) - P_hat_end) norm(K_orig(:,:,end) - K_end) ]

%for i =1 :size(x_orig,1)
%    figure; 
%    subplot(211); plot(x_pred_orig(i,:),x_pred(i,:));
%    subplot(212); plot(x_hat_orig(i,:),x_hat(i,:));            
%end

%for i =1 :size(y_orig,1)
%    figure; plot(y_pred_orig(i,:),y_pred(i,:));            
%end

% clc; [y_pred' y_pred_orig']
 keyboard      
end  
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x,x_pred,x_hat,y_pred,y,P_pred,P_hat,inno_cov,K] = test_kf_sqrt(A,B,C,D,Q,R,S,x_pred_init,P_pred_init,T);

T = T+1;
noise_matrix = [Q,S;S',R];

%Generate Data
[ny,nx] = size(C);
[nx,nu] = size(B);

x = zeros(nx,T+1); 
u = randn(nu,T+1);
y = zeros(ny,T+1);
w = zeros(nx,T+1);
v = zeros(ny,T+1);
wv = zeros(nx+ny,T+1);

for t = 1:T+1
    wv(:,t) = chol(noise_matrix)'*randn(nx+ny,1);
    w(:,t) = wv(1:nx,t);
    v(:,t) = wv(nx+1:end,t);
    
    y(:,t)   = C*x(:,t) + D*u(:,t); +v(:,t); if (t == T+1) break; end
    x(:,t+1) = A*x(:,t) + B*u(:,t) + w(:,t);
    
end

A_eff = A-S*inv(R)*C
Q_eff = Q - S*inv(R)*S'
E     = S*inv(R)

%Kalman Filtering
x_pred = zeros(nx,T+1); P_Pred = zeros(nx,nx,T+1); x_pred(:,1) = x_pred_init; P_pred(:,:,1) = P_pred_init;
x_hat  = zeros(nx,T+1); P_hat  = zeros(nx,nx,T+1);
y_pred = zeros(ny,T+1); inno_cov = zeros(ny,ny,T+1); y_pred(:,1) = C*x_pred(:,1) + D*u(:,1); inno_cov(:,:,1) = C*P_pred(:,:,1)*C'+R;
K = zeros(nx,ny,T+1);

for t = 1:T+1
    
%Measurement Update
PRE = zeros(ny+nx,ny+nx);
PRE(1:ny,1:ny)              = chol(R)';
PRE(1:ny, ny+1:end)         = C*chol(P_pred(:,:,t))'; 
PRE(ny+1:end, ny+1:end)     = chol(P_pred(:,:,t))';
PRE                         = PRE'; 
[dummy,POST]                = qr(PRE);
POST                        = POST.';
P_hat(:,:,t)                = POST(ny+1:end,ny+1:end)*POST(ny+1:end,ny+1:end)';
K(:,:,t)                    = POST(ny+1:end,1:ny)*inv(POST(1:ny,1:ny));

x_hat(:,t)                  = x_pred(:,t) + K(:,:,t)*(y(:,t) - y_pred(:,t) );

if t == T+1
    break;
end

%Time Update 
PRE                     = zeros(nx,2*nx);
PRE(1:nx,1:nx)          = A_eff*chol(P_hat(:,:,t))';
PRE(1:nx,nx+1:end)      = chol(Q_eff)';
PRE                     = PRE';
[dummy,POST]            = qr(PRE);
POST                    = POST';
P_pred(:,:,t+1)         = POST(1:nx,1:nx)*POST(1:nx,1:nx)';
inno_cov(:,:,t+1)       = C*P_pred(:,:,t+1)*C' + R;

x_pred(:,t+1)           = A_eff*x_hat(:,t) + B*u(:,t) + E*y(:,t);
y_pred(:,t+1)           = C*x_pred(:,t+1) + D*u(:,t+1);

end

csvwrite('T_in',T-1);
csvwrite('A_in',A'); 
csvwrite('B_in',B');
csvwrite('C_in',C');
csvwrite('D_in',D');
csvwrite('Q_in',Q');
csvwrite('R_in',R'); 
csvwrite('S_in',S');   
csvwrite('u_in',u');
csvwrite('w_in',w');
csvwrite('v_in',v');
csvwrite('y_in',y');
csvwrite('x_in',x'); 
csvwrite('x_pred_0_in',x_pred(:,1)');
csvwrite('y_pred_0_in',y_pred(:,1)');
csvwrite('P_pred_0_in',P_pred(:,:,1)');
csvwrite('inno_cov_0_in',inno_cov(:,:,1)');
end %end of function test_kf_sqrt

 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x,x_pred,x_hat,y_pred,y,P_pred,P_hat,inno_cov,K] = test_kf_no_sqrt(A,B,C,D,Q,R,S,x_pred_init,P_pred_init,T);
clc; 

T = T+1;
noise_matrix = [Q,S;S',R];

%Generate Data
[ny,nx] = size(C);
[nx,nu] = size(B);

x = zeros(nx,T+1); 
u = randn(nu,T+1);
y = zeros(ny,T+1);
w = zeros(nx,T+1);
v = zeros(ny,T+1);
wv = zeros(nx+ny,T+1);

for t = 1:T+1
    wv(:,t) = chol(noise_matrix)'*randn(nx+ny,1);
    w(:,t) = wv(1:nx,t);
    v(:,t) = wv(nx+1:end,t);
    
    y(:,t)   = C*x(:,t) + D*u(:,t); +v(:,t); if (t == T+1) break; end
    x(:,t+1) = A*x(:,t) + B*u(:,t) + w(:,t);
    
end

A_eff = A-S*inv(R)*C
Q_eff = Q - S*inv(R)*S'
E     = S*inv(R)

%Kalman Filtering
x_pred = zeros(nx,T+1); P_Pred = zeros(nx,nx,T+1); x_pred(:,1) = x_pred_init; P_pred(:,:,1) = P_pred_init;
x_hat  = zeros(nx,T+1); P_hat  = zeros(nx,nx,T+1);
y_pred = zeros(ny,T+1); inno_cov = zeros(ny,ny,T+1); y_pred(:,1) = C*x_pred(:,1) + D*u(:,1); inno_cov(:,:,1) = C*P_pred(:,:,1)*C'+R;
K = zeros(nx,ny,T+1);

for t = 1:T+1
    
%Measurement Update
K(:,:,t)                    = P_pred(:,:,t)*C'*inv(R+C*P_pred(:,:,t)*C');
P_hat(:,:,t)                = (eye(nx,nx) - K(:,:,t)*C)*P_pred(:,:,t);

x_hat(:,t)                  = x_pred(:,t) + K(:,:,t)*(y(:,t) - y_pred(:,t) );

if t == T+1
    break;
end

%Time Update 
P_pred(:,:,t+1) = A_eff*P_hat(:,:,t)*A_eff' + Q_eff;

inno_cov(:,:,t+1)       = C*P_pred(:,:,t+1)*C' + R;

x_pred(:,t+1)           = A_eff*x_hat(:,t) + B*u(:,t) + E*y(:,t);
y_pred(:,t+1)           = C*x_pred(:,t+1) + D*u(:,t+1);

end

csvwrite('T_in',T-1);
csvwrite('A_in',A'); 
csvwrite('B_in',B');
csvwrite('C_in',C');
csvwrite('D_in',D');
csvwrite('Q_in',Q');
csvwrite('R_in',R'); 
csvwrite('S_in',S');   
csvwrite('u_in',u');
csvwrite('w_in',w');
csvwrite('v_in',v');
csvwrite('y_in',y');
csvwrite('x_in',x'); 
csvwrite('x_pred_0',x_pred(:,1)');
csvwrite('y_pred_0',y_pred(:,1)');
csvwrite('P_pred_0',P_pred(:,:,1)');
csvwrite('inno_cov_0',inno_cov(:,:,1)');
end %end of function test_kf_no_sqrt



