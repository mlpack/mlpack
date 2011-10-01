$path = "fx/main_dual";
$dataset_list = `ls $path`;
@datasets = split/\n/, $dataset_list;

foreach $dataset (@datasets) {
    print "$dataset\n";
    $q = -1;
    $r = -1;
    $dim = -1;

    $trial_list = `ls $path/$dataset`;
    @trials = split/\n/, $trial_list;

    my %btable;
    my %qtable;
    foreach $trial (@trials) {
	@a = split/__/, $trial;
	@b = split/_/, $a[0];
	@c = split/_/, $a[1];
	$epsilon = $b[$#b];
	$alpha = $c[$#c];
	$file = "$path/$dataset/$trial/output.txt";
	open FILE, $file;
	while (<FILE>) {
	    chomp($_);
	    @param_vals = split/:/, $_;
	    if ($dim == -1) {
		if ($param_vals[0] eq "/ann/dim") {
		    @temp = split/ /, $param_vals[1];
		    $dim = $temp[1];
		}
	    } elsif ($q == -1) {
		if ($param_vals[0] eq "/ann/qsize") {
		    @temp = split/ /, $param_vals[1];
		    $q = $temp[1];
		}
	    } elsif ($r == -1) {
		if ($param_vals[0] eq "/ann/rsize") {
		    @temp = split/ /, $param_vals[1];
		    $r = $temp[1];
		}
	    } elsif ($param_vals[0] eq "/ann/approx_init/user") {
		@time = split/ /, $param_vals[1];
		$btable { $epsilon }{ $alpha } = $time[1];
	    } elsif ($param_vals[0] eq "/ann/approx/user") {
		@time = split/ /, $param_vals[1];
		$qtable { $epsilon }{ $alpha } = $time[1];
	    }
	}
    }
    print "|Q|: $q, |R|: $r D: $dim\n";
    print "--------------------------------------------------------------\n";
    print "Build time\n";
    print "--------------------------------------------------------------\n";
    $epsilon = 0.1;
    print "alpha\t";
    foreach $alpha (reverse sort keys %{$btable{$epsilon}}) {
	print "$alpha\t";
    }
    print "\n";
    print "epsilon\t";
    foreach $alpha (reverse sort keys %{$btable{$epsilon}}) {
	print "\t";
    }
    print "Rank Error\n";
    foreach $key1 (sort keys %btable ) {
	print "$key1\t";
	foreach $key2 (reverse sort keys %{$btable{$key1}}) {
	    print "$btable{$key1}{$key2}\t";
	}
	$re = $key1*$r/100;
	@wq = split/\./, "$re";
	print "$wq[0]\n";
    }
    print "--------------------------------------------------------------\n";
    print "Query time\n";
    print "--------------------------------------------------------------\n";
    $epsilon = 0.1;
    print "alpha\t";
    foreach $alpha (reverse sort keys %{$qtable{$epsilon}}) {
	print "$alpha\t";
    }
    print "\n";
    print "epsilon\t";
    foreach $alpha (reverse sort keys %{$qtable{$epsilon}}) {
	print "\t";
    }
    print "Rank Error\n";
    foreach $key1 (sort keys %qtable ) {
	print "$key1\t";
	foreach $key2 (reverse sort keys %{$qtable{$key1}}) {
	    print "$qtable{$key1}{$key2}\t";
	}
	$re = $key1*$r/100;
	@wq = split/\./, "$re";
	print "$wq[0]\n";
    }
    print "--------------------------------------------------------------\n";
    print "--------------------------------------------------------------\n";
    print "\n\n";
}
