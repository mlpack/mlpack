%clear all;
function norm_log = loglik(model, M, N)
n_sample = 100;
len_sample = 100:100:1200;
n_data = length(len_sample);

norm_log = zeros(1, n_data);
for i_d=1:length(len_sample)
    len = len_sample(i_d);
    seq = cell(1, n_sample);

    prefix = [num2str(M) '_' num2str(N) '_' num2str(len) '_' num2str(n_sample)];

    [tr, e] = loadHMM(['model/' prefix '_' model '.hmm'], M, N);

    fid = fopen(['data/' prefix '.seq'], 'r');
    for i=1:length(seq)
        str = fgets(fid);
        [tmp_seq,count] = fscanf(fid,'%d,', inf);
        seq{i} = tmp_seq';
    end
    fclose(fid);

    logpseq = zeros(1, n_sample);
    for i=1:length(seq)
        [pStates, tmp_logpseq] = hmmdecode(seq{i},tr,e, 'Symbols', 0:N-1);
        logpseq(i) = tmp_logpseq;
    end
    norm_log(i_d) = mean(logpseq)/len;
end
