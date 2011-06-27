if ($#ARGV != 6) {
    die "Not the right number of arguments";
} else {
    $file_prefix = $ARGV[0];
    $num_classes = $ARGV[1];
    $test_file = $ARGV[2];
    $test_labels_file = $ARGV[3];
    $folds = $ARGV[4];
    $min_leaf = $ARGV[5];
    $max_leaf = $ARGV[6];


    # make a list of training files
    $to = `ls $file_prefix*.csv`;
    @a = split/\n/, $to;
    $p = 0;
    my @train_files;
    foreach $file(@a) {
	if ($file=~/[0-9]/) {
	    push(@train_files, $file);
	    print "$file\n";
	}
    }

    # computing the set sizes to compute the priors
    my @sizes;
    $total_points = 0;
    for ($i = 0; $i < $num_classes; $i++) {
	$to = `wc -l $train_files[$i]`;
	@a = split / /, $to;
	$sizes[$i] = $a[0];
	$total_points += $sizes[$i];
	undef @a;
    }

    print "$total_points training points...\n";
    $to = `wc -l $test_file`;
    @a = split / /, $to;
    $test_size = $a[0];
    undef @a;
    print "$test_size query points....\n";


    # compute the P(Y | X) for all Y and store it as a string
    # for each class
    my @pY_X;

    for ($i = 0; $i < $num_classes; $i++) {

	$p_y = $sizes[$i] / $total_points;
	$dt_out_f = "dt_out_tmp.csv";

	$to = `./dt_driver --data=$train_files[$i] --folds=$folds --test=$test_file --test_output=$dt_out_f --dtree/min_leaf_size=$max_leaf --dtree/max_leaf_size=$max_leaf`;

	#$str_py_x = "";
	open IN, $dt_out_f or die "Cant open $dt_out_f...\n";

	my @py_X;
	while(<IN>) {
	    $py_x = $_ * $p_y;
	    #$str_py_x = $str_py_x." ".$py_x;
	    push(@py_X, $py_x);
	}

	push(@pY_X, \@py_X);

	close IN;
	$to = `rm -rf $dt_out_f`;
    }


    # computing the labels using the posteriors
    my @calc_labels;
    for ($j = 0; $j < $test_size; $j++) {
	$max = -1.0;
	$max_index = -1;
	for ($i = 0; $i < $num_classes; $i++) {
	    if ($pY_X[$i][$j] > $max) {
		$max = $pY_X[$i][$j];
		$max_index = $i;
	    }
	}
	push(@calc_labels, $max_index);
    }

    my @true_labels;
    open IN, $test_labels_file or die "Cant open $test_labels_file..\n";

    while (<IN>) {
	push(@true_labels, $_);
    }

    close IN;

    # checking
    if ($#calc_labels != $#true_labels) {
	die "number of labels not equal..\n";
    }

    $correct = 0;

    for ($i = 0; $i < $#calc_labels; $i++) {
	if ($calc_labels[$i] == $true_labels[$i]) {
	    $correct++;
	}
    }

    $acc = $correct / $test_size;

    print "Accuracy: $acc\n$correct / $test_size\n";
    
}
