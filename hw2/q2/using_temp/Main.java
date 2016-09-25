package q2;

import java.lang.* ; 
import java.util.concurrent.atomic.* ; 
import java.util.concurrent.locks.*  ;

public class Main {

    public static class incThread extends Thread {
        public Counter counter; 
        public int numInc; 
        public int myId; 
        incThread(Counter cntr, int num, int pid) {
          this.counter = cntr; 
          this.numInc  = num; 
          this.myId    = pid; 
        }

        public void run() {
          for ( int i = 0; i < numInc; i++ ) {
            counter.increment(myId); 
          }
        }
    }

    public static void main (String[] args) {
        Counter counter;
        MyLock lock;
        long executeTimeMS = 0;
        int numThread = 6;
        int numTotalInc = 1200000;

        if (args.length < 3) {
            System.err.println("Provide 3 arguments");
            System.err.println("\t(1) <algorithm>: fast/bakery/synchronized/"
                    + "reentrant");
            System.err.println("\t(2) <numThread>: the number of test thread");
            System.err.println("\t(3) <numTotalInc>: the total number of "
                    + "increment operations performed");
            System.exit(-1);
        }

        numThread = Integer.parseInt(args[1]);
        numTotalInc = Integer.parseInt(args[2]);


        if (args[0].equals("fast")) {
            lock    = new FastMutexLock(numThread);
            counter = new LockCounter(lock);
        } else if (args[0].equals("bakery")) {
            lock    = new BakeryLock(numThread);
            counter = new LockCounter(lock);
        } else {
            lock    = new FastMutexLock(numThread);
            counter = new LockCounter(lock);
            System.err.println("ERROR: no such algorithm implemented");
            System.exit(-1);
        }

        // TODO
        // Please create numThread threads to increment the counter
        // Each thread executes numTotalInc/numThread increments
        // Please calculate the total execute time in millisecond and store the
        // result in executeTimeMS
        int numInc = numTotalInc/numThread; 
        incThread [] threadArr = new incThread[numThread]; 
        for (int i = 0; i < numThread; i++) {
          threadArr[i] = new incThread(counter, numInc, i); 
        }

        long startTime = System.currentTimeMillis(); 
        for (int i = 0; i < numThread; i++ ) { 
            threadArr[i].start(); 
        }
        for (int i = 0; i < numThread; i++ ) { 
          try {
            threadArr[i].join(); 
          } catch (InterruptedException e) {} 
        }
        long  endTime = System.currentTimeMillis(); 
        executeTimeMS = endTime - startTime; 

        System.out.println(executeTimeMS);
    }
}
