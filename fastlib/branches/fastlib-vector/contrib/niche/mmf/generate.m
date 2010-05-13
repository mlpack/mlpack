function generate(M, N);

if (M==2 && N == 6) 
    tr = [0.95,0.05; ...
          0.10,0.90];
    e = [1/6 1/6 1/6 1/6 1/6 1/6; ...
         1/10 1/10 1/10 1/10 1/10 1/2];
end

n_sample = 100;
len_sample = 100:100:2000;

for i_d=1:length(len_sample)
    len = len_sample(i_d);
    seq = cell(1, n_sample);

    for i=1:n_sample
        [tmp_seq,states] = hmmgenerate(len,tr,e, 'Symbols', 0:N-1);
        seq{i} = tmp_seq;
    end

    prefix = [num2str(M) '_' num2str(N) '_' num2str(len) '_' num2str(n_sample)];
    fid = fopen(['data/' prefix '.seq'], 'w');
    for i=1:length(seq)
        fprintf(fid,'%% sequence %d\n', i);
        fprintf(fid,'%d,', seq{i});
        fprintf(fid,'\n');
    end
    fclose(fid);

    saveHMM(['model/' prefix '_true.hmm'], tr, e);
end
