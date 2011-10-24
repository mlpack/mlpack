function [testerrs] = test_error_unbalanced(W,testx,testy,task_indexes)

T = length(task_indexes);
testerrs = zeros(T,1);
task_indexes(T+1) = length(testy)+1;
dim = size(testx,1);

for t = 1:T
    t_testx = testx(:, task_indexes(t):task_indexes(t+1)-1 );
    t_testy = testy(task_indexes(t):task_indexes(t+1)-1)';
    prediction = W(:,t)' * t_testx;
    testerrs(t) = (t_testy - prediction) * (t_testy - prediction)' / (task_indexes(t+1)-task_indexes(t));
end

