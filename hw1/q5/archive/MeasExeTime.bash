#!/bin/bash 

echo "#-------------------------------------------------"
echo "#> Using Peterson's Tournament Algorithm" 
echo "#-------------------------------------------------"

echo "## Number of threads: 1"
java PTournamentInc 1
echo "## Number of threads: 2"
java PTournamentInc 2
echo "## Number of threads: 3"
java PTournamentInc 3
echo "## Number of threads: 4"
java PTournamentInc 4
echo "## Number of threads: 5"
java PTournamentInc 5
echo "## Number of threads: 6"
java PTournamentInc 6
echo "## Number of threads: 7"
java PTournamentInc 7
echo "## Number of threads: 8"
java PTournamentInc 8

echo ""

echo "#-------------------------------------------------"
echo "#> Using Java's AtomicInteger" 
echo "#-------------------------------------------------"

echo "## Number of threads: 1"
java AtomicInc 1
echo "## Number of threads: 2"
java AtomicInc 2
echo "## Number of threads: 3"
java AtomicInc 3
echo "## Number of threads: 4"
java AtomicInc 4
echo "## Number of threads: 5"
java AtomicInc 5
echo "## Number of threads: 6"
java AtomicInc 6
echo "## Number of threads: 7"
java AtomicInc 7
echo "## Number of threads: 8"
java AtomicInc 8

echo ""

echo "#-------------------------------------------------"
echo "#> Using Java's synchronized construct" 
echo "#-------------------------------------------------"

echo "## Number of threads: 1"
java SyncInc 1
echo "## Number of threads: 2"
java SyncInc 2
echo "## Number of threads: 3"
java SyncInc 3
echo "## Number of threads: 4"
java SyncInc 4
echo "## Number of threads: 5"
java SyncInc 5
echo "## Number of threads: 6"
java SyncInc 6
echo "## Number of threads: 7"
java SyncInc 7
echo "## Number of threads: 8"
java SyncInc 8

echo ""

echo "# --------------------------------------------------"
echo "#> Using Java's ReentrantLock" 
echo "# --------------------------------------------------"

echo "## Number of threads: 1"
java RLockInc 1
echo "## Number of threads: 2"
java RLockInc 2
echo "## Number of threads: 3"
java RLockInc 3
echo "## Number of threads: 4"
java RLockInc 4
echo "## Number of threads: 5"
java RLockInc 5
echo "## Number of threads: 6"
java RLockInc 6
echo "## Number of threads: 7"
java RLockInc 7
echo "## Number of threads: 8"
java RLockInc 8

echo ""

