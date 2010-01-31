#!/usr/bin/perl -w
# This perl script removes duplicate include paths left to the right
use strict;
my @all_incl_paths = @ARGV;
my @cleaned_up_incl_paths;
foreach( @all_incl_paths ) {
	$_ = remove_rel_paths($_);
	if( !($_=~/-I/) ) {
		push @cleaned_up_incl_paths, $_;
	}
	elsif( !entry_exists($_,\@cleaned_up_incl_paths) ) {
		push @cleaned_up_incl_paths, $_;
	}
}
print join( " ", @cleaned_up_incl_paths );
#
# Subroutines
#
sub entry_exists {
	my $entry = shift; # String
	my $list  = shift; # Reference to an array
	foreach( @$list ) {
		if( $entry eq $_ ) { return 1; }
	}
	return 0;
}
#
sub remove_rel_paths {
	my $entry_in = shift;
	if ($entry_in=~/-I\.\./) {
		return $entry_in;
	}
	my @paths = split("/",$entry_in);
	my @new_paths;
	foreach( @paths ) {
		if( !($_=~/\.\./) ) {
			push @new_paths, $_;
		}
		else {
			pop @new_paths
		}
	}
	return join("/",@new_paths);
}
