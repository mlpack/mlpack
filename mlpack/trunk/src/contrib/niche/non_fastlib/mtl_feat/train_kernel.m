function [W,costfunc,err,reg] = train_kernel(trainx,trainy,task_indexes,gamma,kernel_method)

num_data = size(trainx,2);
dim = size(trainx,1);
T = length(task_indexes);
task_indexes(T+1) = num_data+1;

costfunc = 0;
err = 0;
reg = 0;

for t = 1:T
    % get the data for this task
    x = trainx(: , task_indexes(t):task_indexes(t+1)-1);
    y = trainy(task_indexes(t):task_indexes(t+1)-1);
    K = x'*x;
    [a, costfunct, errt, regt] = feval(kernel_method,K,y,gamma);
    W(:,t) = x*a;

    costfunc = costfunc + costfunct;
    err = err + errt;
    reg = reg + regt;
end

