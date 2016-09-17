import java.lang.* ; 
import java.util.concurrent.locks.* ; 
import java.util.concurrent.atomic.* ; 
import java.util.concurrent.locks.ReentrantLock; 

public class q5 {
  
  /* Lock based on Peterson Tournament Algorithm */
  static class PTournamentLock extends ReentrantLock { 
  
      /* Number of threads */
      private int n; 
  
      /* The gate number each thread is at: gate[thread#] */
      /* Gates and levels are numbered from root to leafs */
      private int [] gate; 
      /* The stuck thread at each gate: last[gate#] */
      private AtomicInteger [] last; 
  
      /* Get the total number of gates */
      private int getNumGate() {
         return n - 1; // Only consider 1, 2, 4 or 8 threads
      }

      private double log2(double x) {
        return Math.log10(x)/Math.log10(2.0); 
      }
  
      /* CTOR */
      public PTournamentLock(int n_val) {
          n = n_val; 
          gate  = new int[n + 1]; 
          last  = new AtomicInteger[getNumGate() + 1]; 

          for (int i = 0; i < last.length; i++ ) 
            last[i] = new AtomicInteger(0); 
      }
  
      /* The i-th thread    */
      /* i = 1: n           */
      /* k = 1: gate.length */
      public void lock(int pid) {
        int i = pid + 1; 
        int k = last.length - (i + 1)/2; 
  
        while ( k > 0 ) {
          gate[i] = k; 
          last[k].getAndSet(i); 
  
          /* The smallest gate number in the current level */
          int check_gate = (int) Math.pow(2, Math.floor(log2((double)k))); 
  
          for ( int j = 1; j < n + 1; j++ ) 
            while ( j != i && last[k].get() == i && (gate[j] < check_gate || gate[j] == k));   
          // while (last[k].get() == i); 
          // System.out.println("PID " + i + " gets through") ;
  
          // last[k].getAndSet(0); 
          k = k/2; 
        } // while ( k > 0 )
  
      } // public void lock()
  
      public void unlock(int pid) {
          gate[pid + 1] = Integer.MAX_VALUE; 
      }
  }

  /* The incrementing counter implementation based on Tournament Algorithm */
  static class PTournamentInc implements Runnable { 
      public static volatile int c = 0; 
      public int pid; 
      public int m; 
      public int n; 
      private static PTournamentLock lock; 
      private int numIncr; 
      public PTournamentInc(int m_val, int n_val, int pid_val) { 
          m = m_val; 
          n = n_val; 
          numIncr = (int) Math.ceil(m/(double)n);
          pid = pid_val; 
          lock = new PTournamentLock(n); 
      }
      public void run() { 
          for ( int i = 0; i < numIncr; i++ ) {
            lock.lock(pid); 
            c++; 
            lock.unlock(pid); 
          }
      }
      public static void main_a(int n) { 
          int m = 1200000; 
  
          PTournamentInc [] f_array; 
          Thread [] t_array; 
  
          f_array = new PTournamentInc[n]; 
          t_array = new Thread[n]; 
  
          for (int i = 0; i < n; i++ ) { 
              f_array[i] = new PTournamentInc(m, n, i); 
              t_array[i] = new Thread(f_array[i]); 
          }
  
          long startTime = System.currentTimeMillis(); 
  
          for (int i = 0; i < n; i++ ) { 
            t_array[i].start(); 
          }

          for (int i = 0; i < n; i++ ) { 
            try {
              t_array[i].join(); 
            } catch (InterruptedException e) {} 
          }
  
          long endTime = System.currentTimeMillis(); 
          System.out.println("With " + n + " threads: " + (endTime - startTime) + " ms / final counter value: " + c); 
      }
  }
  
  /* The incrementing counter based on AtomicInteger */ 
  static class AtomicInc implements Runnable { 
      public static volatile AtomicInteger c = new AtomicInteger(); 
      public int m; 
      public int n; 
      public AtomicInc(int m_val, int n_val) { 
          m = m_val; 
          n = n_val; 
      }
      public void run() { 
          try {
            for ( int i = 0; i < Math.ceil(m/(double)n); i++ ) {
                c.getAndAdd(1); 
            }
          } finally {}
      }
      public static void main_b(int n_thread) { 
          int m = 1200000; 
          int n = n_thread; 
  
          AtomicInc [] f_array; 
          Thread [] t_array; 
  
          f_array = new AtomicInc[n]; 
          t_array = new Thread[n]; 
  
          for (int i = 0; i < n; i++ ) { 
              f_array[i] = new AtomicInc(m, n); 
              t_array[i] = new Thread(f_array[i]); 
          }
  
          long startTime = System.currentTimeMillis(); 
  
          for (int i = 0; i < n; i++ ) { 
              t_array[i].start(); 
          }
  
          for (int i = 0; i < n; i++ ) { 
            try {
              t_array[i].join(); 
            } catch (InterruptedException e) {} 
          }
  
          long endTime = System.currentTimeMillis(); 
          System.out.println("With " + n + " threads: " + (endTime - startTime) + " ms / final counter value: " + c); 
      }
  }

  /* Syncrhonized counter */
  static class SyncCnt {
      private int cnt = 0; 
      public synchronized void inc() {
          cnt++; 
      }
      public synchronized void dec() {
          cnt--; 
      }
      public int get() { 
          return cnt; 
      }
      public void rst() { 
          cnt = 0; 
      }
  }


  /* The incrementing counter based on the synchronized counter */
  static class SyncInc implements Runnable { 
      public boolean complete = true; 
      public static SyncCnt c = new SyncCnt(); 
      public int m; 
      public int n; 
      public SyncInc(int m_val, int n_val) { 
          m = m_val; 
          n = n_val; 
      }
      public void run() { 
          complete = false; 
          for ( int i = 0; i < Math.ceil(m/(double)n); i++ ) {
              c.inc(); 
          }
          complete = true; 
      }
      public static void main_c(int n_thread) { 
          int m = 1200000; 
          int n = n_thread; 
  
          SyncInc [] f_array; 
          Thread [] t_array; 
  
          f_array = new SyncInc[n]; 
          t_array = new Thread[n]; 
  
          for (int i = 0; i < n; i++ ) { 
              f_array[i] = new SyncInc(m, n); 
              t_array[i] = new Thread(f_array[i]); 
          }
  
          long startTime = System.currentTimeMillis(); 
  
          for (int i = 0; i < n; i++ ) { 
              t_array[i].start(); 
          }
  
          for (int i = 0; i < n; i++ ) { 
            try {
              t_array[i].join(); 
            } catch (InterruptedException e) {} 
          }
  
          long endTime = System.currentTimeMillis(); 
          System.out.println("With " + n + " threads: " + (endTime - startTime) + " ms / final counter value: " + c.get()); 
      }
  }

  /* The incrementing counter based on the Reentrant Lock */
  static class RLockInc implements Runnable { 
      public boolean complete = true; 
      private static ReentrantLock lock = new ReentrantLock(); 
      public static volatile int c = 0; 
      public int m; 
      public int n; 
      public RLockInc(int m_val, int n_val) { 
          m = m_val; 
          n = n_val; 
      }
      public void run() { 
          complete = false; 
          lock.lock(); 
          try { 
              for ( int i = 0; i < Math.ceil(m/(double)n); i++ ) {
                  c++; 
              }
          } finally { 
              lock.unlock(); 
          }
          complete = true; 
      }
      public static void main_d(int n_thread) { 
          int m = 1200000; 
          int n = n_thread; 
  
          RLockInc [] f_array; 
          Thread [] t_array; 
  
          f_array = new RLockInc[n]; 
          t_array = new Thread[n]; 
  
          for (int i = 0; i < n; i++ ) { 
              f_array[i] = new RLockInc(m, n); 
              t_array[i] = new Thread(f_array[i]); 
          }
  
          long startTime = System.currentTimeMillis(); 
  
          for (int i = 0; i < n; i++ ) { 
              t_array[i].start(); 
          }
  
          for (int i = 0; i < n; i++ ) { 
            try {
              t_array[i].join(); 
            } catch (InterruptedException e) {} 
          }
  
          long endTime = System.currentTimeMillis(); 
          System.out.println("With " + n + " threads: " + (endTime - startTime) + " ms / final counter value: " + c); 
      }
  }

  public static void main(String[] args) {
    /* (a) */ 
    System.out.println( "#-------------------------------------------------"); 
    System.out.println( "#> (a) Using Peterson's Tournament Algorithm" );
    System.out.println( "#-------------------------------------------------");

    int [] numThreads = new int[]{1, 2, 4, 8}; 
    for ( int i : numThreads ) {
      PTournamentInc.c = 0; 
      PTournamentInc.main_a(i); 
    }

    /* (b) */ 
    System.out.println( "#-------------------------------------------------"); 
    System.out.println( "#> (b) Using Java's AtomicInteger" );
    System.out.println( "#-------------------------------------------------");

    for ( int i = 1; i < 9; i++ ) {
      AtomicInc.c.getAndSet(0); 
      AtomicInc.main_b(i); 
    }

    /* (c) */ 
    System.out.println( "#-------------------------------------------------"); 
    System.out.println( "#> (c) Using Java's synchronized construct" );
    System.out.println( "#-------------------------------------------------");

    for ( int i = 1; i < 9; i++ ) {
      SyncInc.c.rst(); 
      SyncInc.main_c(i); 
    }

    /* (d) */ 
    System.out.println( "#-------------------------------------------------"); 
    System.out.println( "#> (d) Using Java's ReentrantLock" );
    System.out.println( "#-------------------------------------------------");

    for ( int i = 1; i < 9; i++ ) {
      RLockInc.c = 0; 
      RLockInc.main_d(i); 
    }

  }
      
}
