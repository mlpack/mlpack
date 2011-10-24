function [w,gamma,trainCorr, testCorr, cpu_time, nu]=lpsvm(A,d,k,nu,output,delta)
% version 1.3
% last revision: 07/07/03
%===========================================================================
% Usage: [w, gamma,trainCorr,testCorr,time,nu]=lpsvm(A,d,k,nu,output,delta);
%
% A and d are both required, everything else has a default
% An example: [w gamma train test time nu] = lpsvm(A,d,10);
%
% Input parameters:
%    A: Data points
%    d: 1's or -1's
%    k: way to divide the data set into test and training set
%       if k = 0: simply run the algorithm without any correctness
%         calculation, this is the default
%       if k = 1: run the algorithm and calculate correctness on
%         the whole data set
%       if k = any value less than the # of rows in the data set:
%         divide up the data set into test and training
%         using k-fold method
%       if k = # of rows in the data set: use the 'leave 1' method
%
%    output: 0 - no output, 1 - produce output, default is 0
%    nu:             weighted parameter
%                    -1 - easy estimation
%                    0  - hard estimation
%                    any other value - used as nu by the algorithm
%                    default - 0
%    delta:  default is 10^-3
%===================================================================
% Output parameters:
%
%       w:              the normal vector of the classifier
%       gamma:          the threshold
%       trainCorr:      training set correctness
%       testCorr:       test set correctness
%       cpu_time:       time elapsed
%       nu:             estimated value (or specified value) of nu
%==========================================================================

if nargin<6
delta=1e-3;
end

if nargin<5
output=0;
end

if ((nargin<4)|(nu==0))
     nu = EstNuLong(A,d);  % default is hard estimation
elseif nu==-1  % easy estimation
nu = EstNuShort(A,d);
end

if nargin<3
k=0;
end

r=randperm(size(d,1));d=d(r,:);A=A(r,:);    % random permutation

tic;

trainCorr=0;
testCorr=0;

if k==0
[w, gamma,iter] = core(A,d,nu,delta);
cpu_time=toc;
  if output==1
fprintf(1,'\nNumber of Iterations: %d',iter);
fprintf(1,'\nElapse time: %10.2f\n\n',cpu_time);
  end
  return
end

%if k==1 only training set correctness is calculated
if k==1
tic;
[w, gamma,iter] = core(A,d,nu,delta);
trainCorr = correctness(A,d,w,gamma);
cpu_time = toc;
  if output == 1
fprintf(1,'\nTraining set correctness: %3.2f%%',trainCorr);
fprintf(1,'\nNumber of Iterations: %d',iter);
fprintf(1,'\nElapse time: %10.2f\n\n',cpu_time);
  end
  return
end

[sm sn]=size(A);
accuIter = 0;

cpu_time = 0;
indx = [0:k];
indx = floor(sm*indx/k);    %last row numbers for all 'segments'
% split trainining set from test set
for i = 1:k
Ctest = []; dtest = [];Ctrain = []; dtrain = [];

Ctest = A((indx(i)+1:indx(i+1)),:);
dtest = d(indx(i)+1:indx(i+1));

Ctrain = A(1:indx(i),:);
Ctrain = [Ctrain;A(indx(i+1)+1:sm,:)];
dtrain = [d(1:indx(i));d(indx(i+1)+1:sm,:)];
tic;
[w, gamma,iter] = core(Ctrain,dtrain,nu,delta);
thisToc = toc;

tmpTrainCorr = correctness(Ctrain,dtrain,w,gamma);
tmpTestCorr = correctness(Ctest,dtest,w,gamma);

 if output==1
fprintf(1,'________________________________________________\n');
fprintf(1,'Fold %d\n',i);
fprintf(1,'Training set correctness: %3.2f%%\n',tmpTrainCorr);
fprintf(1,'Testing set correctness: %3.2f%%\n',tmpTestCorr);
fprintf(1,'Number of iterations: %d\n',iter);
fprintf(1,'Elapse time: %10.2f\n',thisToc);
end

trainCorr = trainCorr + tmpTrainCorr;
testCorr = testCorr + tmpTestCorr;
accuIter = accuIter + iter; % accumulative iterations
cpu_time = cpu_time + thisToc;

end % end of for (looping through test sets)

     trainCorr = trainCorr/k;
     testCorr = testCorr/k;
     cpu_time=cpu_time/k;

if output == 1
     fprintf(1,'==============================================');
fprintf(1,'\nTraining set correctness: %3.2f%%',trainCorr);
fprintf(1,'\nTesting set correctness: %3.2f%%',testCorr);
fprintf(1,'\nAverage number of iterations: %d',accuIter/k);
fprintf(1,'\nAverage cpu_time: %10.2f\n',cpu_time);
end

return;  % lpsvm function return

%%%%%%%%%%% core calculation function %%%%%%%%%%%%%%%%%%%%%
function [w, gamma, iter] = core(A,d,nu,delta);

[m,n]=size(A);

if m>=n
[w,gamma,iter]=lpsvm_with_smw(A,d,nu,delta);
else
[w,gamma,iter]=lpsvm_without_smw(A,d,nu,delta);
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   lpsvm when m>=n                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [w,gamma,iter]=lpsvm_with_smw(A,d,nu,delta)
%with SMW without armijo
%parameters
epsi=10^(-3);alpha=10^(2);tol=10^(-5);maxiter=1000;
[m,n]=size(A);
en=ones(n,1);
em=ones(m,1);

% initial u
u=ones(m,1);iter=0;
epsi=epsi*em;nu=nu*em;
diff=1;
DA=spdiags(d,0,m,m)*A;
while (diff>tol) & (iter<maxiter)
    uold=u;
    iter=iter+1;
    du=d.*u;Adu=A'*du;
    pp=max(Adu-en,0);np=max(-Adu-en,0);
    dd=sum(du)*d;unu=max(u-nu,0);uu=max(-u,0);
    %Gradient
    g=-epsi+(d.*(A*pp))-(d.*(A*np))+dd+unu-alpha*uu;
    %Hessian
    E=spdiags(sqrt(sign(np)+sign(pp)),0,n,n);
    H=[DA*E d];
    f=1./(delta+sign(unu)+alpha*sign(uu));
    F=spdiags(f,0,m,m);gg=f.*g;HT=H';
    di=(eye(n+1)+HT*(F*H))\(HT*gg);
    di=H*di;di=f.*di;di=-gg+di;u=u+di;
    diff=norm(g);
end

w=1/epsi(1)*(pp-np);
gamma=-(1/epsi(1))*sum(du);
iter; % semi-colon added by niche
return



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   lpsvm when m<n                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w,gamma,iter]=lpsvm_without_smw(A,d,nu,delta)
%without sherman and without armijo
%parameters
epsi=10^(-1);alpha=10^3;tol=10^(-3);maxiter=50;
[m,n]=size(A);
en=ones(n,1);
em=ones(m,1);
%initial u
u=ones(m,1);iter=0;
epsi=epsi*em;nu=nu*em;
diff=1;
DA=spdiags(d,0,m,m)*A;
while (diff>tol) & (iter<maxiter)
    uold=u;
    iter=iter+1;
    du=d.*u;Adu=A'*du;
    pp=max(Adu-en,0);np=max(-Adu-en,0);
    dd=sum(du)*d;unu=max(u-nu,0);uu=max(-u,0);
    %Gradient
    g=-epsi+(d.*(A*pp))-(d.*(A*np))+dd+unu-alpha*uu;
    %Hessian
    E=spdiags(sqrt(sign(np)+sign(pp)),0,n,n);
    H=[DA*E d];
    F=spdiags(delta+sign(unu)+alpha*sign(uu),0,m,m);
    di=-((H*H'+F)\g);
    u=u+di;
    diff=norm(g);
end
du=d.*u;Adu=A'*du;
pp=max(Adu-en,0);np=max(-Adu-en,0);
w=1/epsi(1)*(pp-np);gamma=-(1/epsi(1))*sum(du);
return

%%%%%%%%%%%%%%%% correctness calculation %%%%%%%%%%%%%%%%

function corr = correctness(AA,dd,w,gamma)

p=sign(AA*w-gamma);
corr=length(find(p==dd))/size(AA,1)*100;
return

%%%%%%%%%%%%%%EstNuLong%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% hard way to estimate nu if not specified by the user
function value = EstNuLong(C,d)

[m,n]=size(C);e=ones(m,1);
H=[C -e];
if m<201
H2=H;d2=d;
else
r=rand(m,1);
[s1,s2]=sort(r);
H2=H(s2(1:200),:);
d2=d(s2(1:200));
end

lamda=1;
[vu,u]=eig(H2*H2');u=diag(u);p=length(u);
yt=d2'*vu;
lamdaO=lamda+1;

cnt=0;
while (abs(lamdaO-lamda)>10e-4) &(cnt<100)
     cnt=cnt+1;
     nu1=0;pr=0;ee=0;waw=0;
     lamdaO=lamda;
     for i=1:p
     nu1= nu1 + lamda/(u(i)+lamda);
pr= pr + u(i)/(u(i)+lamda)^2;
ee= ee + u(i)*yt(i)^2/(u(i)+lamda)^3;
waw= waw + lamda^2*yt(i)^2/(u(i)+lamda)^2;
   end
lamda=nu1*ee/(pr*waw);
end

value = lamda;
if cnt==100
    value=1;
end

return

%%%%%%%%%%%%%%%%%EstNuShort%%%%%%%%%%%%%%%%%%%%%%%

% easy way to estimate nu if not specified by the user
function value = EstNuShort(C,d)

value = 1/(sum(sum(C.^2))/size(C,2));
return
