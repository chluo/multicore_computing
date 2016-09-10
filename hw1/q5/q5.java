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
          /* 
           * For power-of-2 number of threads, 
           * the number of gates is simply (n - 1). 
           * For other number of threads, 
           * increase the number to the nearest even number m, 
           * and then the number of gates m. 
           */
           int    m = (n % 2 != 0)? n + 1 : n; 
           double l = Math.log10((double)m) / Math.log10(2.0);
           return l == Math.floor(l) ? m - 1: m; 
      }
  
      /* CTOR */
      public PTournamentLock(int n_val) {
          n = n_val; 
          gate  = new int[n + 1]; 
          last  = new AtomicInteger[getNumGate() + 1]; 
      }
  
      /* The i-th thread */
      /* i = 1: n */
      /* k = 1: gate.length */
      public void lock(int pid) {
        int i = pid + 1; 
        int k = last.length - (i + 1)/2; 
  
        while ( k > 0 ) {
          gate[i] = k; 
          last[k] = new AtomicInteger(i); 
          // System.out.println("Thread " + i + " is now at gate " + k + " . "); 
  
          /* Check if there is anyone ahead in the thread's path */
          boolean someone_ahead = true; 
  
          /* The smallest gate number in the current level */
          int check_gate = (int) Math.pow(2.0, Math.floor(Math.log10(k)/Math.log10(2.0)));  
          // System.out.println("Thread " + i + " is now checking gate " + check_gate + " . "); 
  
          while ( someone_ahead && last[k].get() == i ) {
            someone_ahead = false;
  
            for ( int j = 1; j < n + 1; j++ ) {
              if ( j != i && (gate[j] < check_gate || gate[j] == k) && gate[j] != 0 )  {
                  someone_ahead = true;
                  break; 
              } // if ( j != i && gate[j] < check_gate )
            } // for ( int j = 1; j < n + 1; j++ )
          } // while ( someone_ahead && last[k] == i )
  
          k = k/2; 
        } // while ( k > 0 )
  
        // System.out.println("Thread " + i + " is now entering CS. "); 
      } // public void lock()
  
      public void unlock(int pid) {
          int i = pid + 1;
          gate[i] = 0; 
          // System.out.println("Thread " + i + " is now leaving CS. "); 
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
  }

  /* The incrementing counter implementation based on Tournament Algorithm */
  static class PTournamentInc implements Runnable { 
      public static volatile int c = 0; 
      public int pid; 
      public int m; 
      public int n; 
      private static PTournamentLock lock; 
      public PTournamentInc(int m_val, int n_val, int pid_val) { 
          m = m_val; 
          n = n_val; 
          pid = pid_val; 
          lock = new PTournamentLock(n); 
      }
      public void run() { 
          lock.lock(pid); 
          try {
            for ( int i = 0; i < Math.ceil(m/(double)n); i++ ) {
                c++; 
            }
          } finally {
            lock.unlock(pid); 
          }
          System.out.println("Current counter value: " + c); 
      }
      public static void main_a(int n_thread) { 
          int m = 1200000; 
          int n = n_thread; 
  
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
  
          boolean someone_alive = true; 
          while (someone_alive ) {
              someone_alive = false; 
              for (int i = 0; i < n; i++ ) { 
                  if (t_array[i].getState() != Thread.State.TERMINATED) 
                      someone_alive = true; 
              }
          }
  
          long endTime = System.currentTimeMillis(); 
          System.out.println("Execution time: " + (endTime - startTime) + " ms"); 
      }
  }
  
  /* The incrementing counter based on AtomicInteger */ 
  static class AtomicInc implements Runnable { 
      public static volatile AtomicInteger c = new AtomicInteger(); 
      /*
      public static int expect = 0; 
      */
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
                /*
                c.compareAndSet(expect, expect + 1); 
                expect++; 
                */
            }
          } finally {
            System.out.println("Current counter value: " + c.get()); 
          }
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
  
          boolean someone_alive = true; 
          while (someone_alive ) {
              someone_alive = false; 
              for (int i = 0; i < n; i++ ) { 
                  if (t_array[i].getState() != Thread.State.TERMINATED) 
                      someone_alive = true; 
              }
          }
  
          long endTime = System.currentTimeMillis(); 
          System.out.println("Execution time: " + (endTime - startTime) + " ms"); 
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
          System.out.println("Current counter value: " + c.get()); 
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
  
          boolean someone_alive = true; 
          while (someone_alive ) {
              someone_alive = false; 
              for (int i = 0; i < n; i++ ) { 
                  if (t_array[i].getState() != Thread.State.TERMINATED) 
                      someone_alive = true; 
              }
          }
  
          long endTime = System.currentTimeMillis(); 
          System.out.println("Execution time: " + (endTime - startTime) + " ms"); 
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
          System.out.println("Current counter value: " + c); 
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
  
          boolean someone_alive = true; 
          while (someone_alive ) {
              someone_alive = false; 
              for (int i = 0; i < n; i++ ) { 
                  if (t_array[i].getState() != Thread.State.TERMINATED) 
                      someone_alive = true; 
              }
          }
  
          long endTime = System.currentTimeMillis(); 
          System.out.println("Execution time: " + (endTime - startTime) + " ms"); 
      }
  }

  public static void main(String[] args) {
    /* (a) */ 
    System.out.println( "#-------------------------------------------------"); 
    System.out.println( "#> (a) Using Peterson's Tournament Algorithm" );
    System.out.println( "#-------------------------------------------------");

    for ( int i = 1; i < 9; i++ ) {
      PTournamentInc.main_a(i); 
    }

    /* (b) */ 
    System.out.println( "#-------------------------------------------------"); 
    System.out.println( "#> (b) Using Java's AtomicInteger" );
    System.out.println( "#-------------------------------------------------");

    for ( int i = 1; i < 9; i++ ) {
      AtomicInc.main_b(i); 
    }

    /* (c) */ 
    System.out.println( "#-------------------------------------------------"); 
    System.out.println( "#> (c) Using Java's synchronized construct" );
    System.out.println( "#-------------------------------------------------");

    for ( int i = 1; i < 9; i++ ) {
      SyncInc.main_c(i); 
    }

    /* (d) */ 
    System.out.println( "#-------------------------------------------------"); 
    System.out.println( "#> (d) Using Java's ReentrantLock" );
    System.out.println( "#-------------------------------------------------");

    for ( int i = 1; i < 9; i++ ) {
      RLockInc.main_d(i); 
    }

  }
      
}
