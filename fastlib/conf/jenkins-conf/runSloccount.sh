#!/bin/bash

if [ ! -d ".slocdata" ]
then
  mkdir .slocdata
fi
sloccount --wide --details --datadir .slocdata repo > .slocdata/sloccount.sc
