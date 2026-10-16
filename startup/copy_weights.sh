#!/bin/bash
# Parallel copy of /prefill_model (ceph-fuse) into /dev/shm; one cp per file, 16 at a time.
SRC=/prefill_model; DST=/dev/shm/prefill_model
mkdir -p $DST
date +%s > /dev/shm/cp_start
ls -1 $SRC | xargs -P 16 -I{} cp -f $SRC/{} $DST/{}
rc=$?
date +%s > /dev/shm/cp_done
echo rc=$rc
