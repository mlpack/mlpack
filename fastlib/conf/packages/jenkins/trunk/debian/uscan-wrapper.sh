#!/bin/bash
# Test the upstream jenkins mirror for war and plugin updates.
#
# Copyright 2011  Sterling Peet <sterling.peet@gatech.edu>

uscan --report-status

if [ $? -eq 0 ]
then
  echo "[uscan-wrapper] uscan reports updates upstream"
else
  echo "[uscan-wrapper] uscan reports local package is up-to-date"
fi
  
