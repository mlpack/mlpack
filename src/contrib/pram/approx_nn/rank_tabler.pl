$path = "fx/main_dual";
$dataset_list = `ls $path`;
@datasets = split/\n/, $dataset_list;

foreach $dataset (@datasets) {
    @dnames = split/_/, $dataset;
    if ($dnames[1] eq "rank") {
	print "$dataset\n";
	$q = -1;
	$r = -1;
	$dim = -1;

	$trial_list = `ls $path/$dataset`;
	@trials = split/\n/, $trial_list;
	
	my %btable;
	my %qtable;
	my %rtable;
	my %etable;
	my %ptable;
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
	    close FILE;

	    $file = "$path/$dataset/$trial/log.txt";
	    open FILE, $file;
	    while (<FILE>) {
		chomp($_);
		@param_vals = split/0m/, $_;
		@binfo = split/,/, $param_vals[1];
		if ($#binfo == 2) {
		    foreach $info(@binfo) {
			#print "$info\n";
			@ainfo = split/:/, $info;
			@cinfo = split/ = /, $info;
			if ($ainfo[0] eq " True Avg Rank error") {
			    $rtable { $epsilon }{ $alpha } = $ainfo[1];
			} elsif ($cinfo[0] eq " True success prob") {
			    $ptable { $epsilon }{ $alpha } = $cinfo[1];
			} elsif ($cinfo[0] eq " Avg de") {
			    $etable { $epsilon }{ $alpha } = $cinfo[1];
			}
		    }
		} elsif ($#binfo == 1) {
		    foreach $info(@binfo) {
			@ainfo = split/: /,$info;
			if ($ainfo[0] eq " Max error") {
			    $maxtable{$epsilon}{$alpha} = $ainfo[1];
			} elsif ($ainfo[0] eq " Min error") {
			    $mintable{$epsilon}{$alpha} = $ainfo[1];
			}
		    }
		}
		undef @binfo;
	    }
	    close FILE;

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
	print "--------------------------------------------------------------\n";
	print "Avg Rank Error\n";
	print "--------------------------------------------------------------\n";
	$epsilon = 0.1;
	print "alpha\t";
	foreach $alpha (reverse sort keys %{$rtable{$epsilon}}) {
	    print "$alpha\t";
	}
	print "\n";
	print "epsilon\t";
	foreach $alpha (reverse sort keys %{$rtable{$epsilon}}) {
	    print "\t";
	}
	print "Rank Error\n";
	foreach $key1 (sort keys %rtable ) {
	    print "$key1\t";
	    foreach $key2 (reverse sort keys %{$rtable{$key1}}) {
		print "$rtable{$key1}{$key2}\t";
	    }
	    $re = $key1*$r/100;
	    @wq = split/\./, "$re";
	    print "$wq[0]\n";
	}
	print "--------------------------------------------------------------\n";
	print "--------------------------------------------------------------\n";
	print "\n\n";
	print "--------------------------------------------------------------\n";
	print "Average Relative Distance Error\n";
	print "--------------------------------------------------------------\n";
	$epsilon = 0.1;
	print "alpha\t";
	foreach $alpha (reverse sort keys %{$etable{$epsilon}}) {
	    print "$alpha\t";
	}
	print "\n";
	print "epsilon\t";
	foreach $alpha (reverse sort keys %{$etable{$epsilon}}) {
	    print "\t";
	}
	print "Rank Error\n";
	foreach $key1 (sort keys %etable ) {
	    print "$key1\t";
	    foreach $key2 (reverse sort keys %{$etable{$key1}}) {
		print "$etable{$key1}{$key2}\t";
	    }
	    $re = $key1*$r/100;
	    @wq = split/\./, "$re";
	    print "$wq[0]\n";
	}
	print "--------------------------------------------------------------\n";
	print "--------------------------------------------------------------\n";
	print "\n\n";
	print "--------------------------------------------------------------\n";
	print "Success probability\n";
	print "--------------------------------------------------------------\n";
	$epsilon = 0.1;
	print "alpha\t";
	foreach $alpha (reverse sort keys %{$ptable{$epsilon}}) {
	    print "$alpha\t";
	}
	print "\n";
	print "epsilon\t";
	foreach $alpha (reverse sort keys %{$ptable{$epsilon}}) {
	    print "\t";
	}
	print "Rank Error\n";
	foreach $key1 (sort keys %ptable ) {
	    print "$key1\t";
	    foreach $key2 (reverse sort keys %{$ptable{$key1}}) {
		print "$ptable{$key1}{$key2}\t";
	    }
	    $re = $key1*$r/100;
	    @wq = split/\./, "$re";
	    print "$wq[0]\n";
	}
	print "--------------------------------------------------------------\n";
	print "--------------------------------------------------------------\n";
	print "\n\n";
	print "Max Rank Error\n";
	print "--------------------------------------------------------------\n";
	$epsilon = 0.1;
	print "alpha\t";
	foreach $alpha (reverse sort keys %{$maxtable{$epsilon}}) {
	    print "$alpha\t";
	}
	print "\n";
	print "epsilon\t";
	foreach $alpha (reverse sort keys %{$maxtable{$epsilon}}) {
	    print "\t";
	}
	print "Rank Error\n";
	foreach $key1 (sort keys %maxtable ) {
	    print "$key1\t";
	    foreach $key2 (reverse sort keys %{$maxtable{$key1}}) {
		print "$maxtable{$key1}{$key2}\t";
	    }
	    $re = $key1*$r/100;
	    @wq = split/\./, "$re";
	    print "$wq[0]\n";
	}
	print "--------------------------------------------------------------\n";
	print "--------------------------------------------------------------\n";
	print "\n\n";
	print "Min Rank Error\n";
	print "--------------------------------------------------------------\n";
	$epsilon = 0.1;
	print "alpha\t";
	foreach $alpha (reverse sort keys %{$mintable{$epsilon}}) {
	    print "$alpha\t";
	}
	print "\n";
	print "epsilon\t";
	foreach $alpha (reverse sort keys %{$mintable{$epsilon}}) {
	    print "\t";
	}
	print "Rank Error\n";
	foreach $key1 (sort keys %mintable ) {
	    print "$key1\t";
	    foreach $key2 (reverse sort keys %{$mintable{$key1}}) {
		print "$mintable{$key1}{$key2}\t";
	    }
	    $re = $key1*$r/100;
	    @wq = split/\./, "$re";
	    print "$wq[0]\n";
	}
	print "--------------------------------------------------------------\n";
	print "--------------------------------------------------------------\n";
	print "\n\n";
    }
}
