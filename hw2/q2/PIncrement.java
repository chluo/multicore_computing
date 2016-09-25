import java.lang.* ; 
import java.util.concurrent.atomic.* ; 
import java.util.concurrent.locks.*  ;

public class PIncrement {

  /*
   * Lamport's Fast Mutex Algorithm
   */
  static class FastMutex { 
    private int n; 
    private volatile int x = -1; 
    private volatile int y = -1; 
    private AtomicInteger [] flag; 

    public FastMutex(int n_that) {
      this.n    = n_that; 
      this.flag = new AtomicInteger[n]; 
      for (int i = 0; i < n; i++) {
        flag[i] = new AtomicInteger(0); 
      }
    }

    public void lock(int pid) {
      while(true) { 
        flag[pid].set(1); 
        x = pid; 
        if (y != -1) {
          flag[pid].set(0); 
          while (y != -1); 
          continue; 
        }
        else {
          y = pid; 
          if (x == pid) {
            // System.out.println("PID " + pid + " enters CS through the fast path"); 
            return; 
          }
          else {
            flag[pid].set(0); 
            for (int j = 0; j < flag.length; j++) 
              while (flag[j].get() == 1); 
            if (y == pid) {
              // System.out.println("PID " + pid + " enters CS through the slow path"); 
              return; 
            }
            else {
              while (y != -1); 
              continue; 
            }
          }
        }
      }
    }

    public void unlock(int pid) {
      // System.out.println("PID " + pid + " leaves CS"); 
      y = -1; 
      flag[pid].set(0); 
    }
  }

  static class FastMutexInc implements Runnable { 
      private static FastMutex lock; 
      public static volatile int c = 0; 
      public int m; 
      public int n; 
      public int pid; 

      public FastMutexInc(int m_that, int n_that, int pid_that) { 
          this.m    = m_that; 
          this.n    = n_that; 
          this.pid  = pid_that; 
      }

      public static void newLock(int n_val) {
        FastMutexInc.lock = new FastMutex(n_val); 
      }

      public void run() { 
        for ( int i = 0; i < Math.ceil(m/(double)n); i++ ) {
          lock.lock(pid); 
          try {
            c++; 
          } finally {
            lock.unlock(pid); 
          }
        }
      }

      public static void doit(int n_thread) { 
          int m = 1200000; 
          int n = n_thread; 
  
          FastMutexInc [] f_array = new FastMutexInc[n]; 
          Thread       [] t_array = new Thread[n]; 

          FastMutexInc.newLock(n); 
  
          for (int i = 0; i < n; i++ ) { 
              f_array[i] = new FastMutexInc(m, n, i); 
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
          long exeTime = endTime - startTime; 
          System.out.println("With " + n + " threads: " + exeTime + " ms / final counter value: " + c); 
      }
  }

  /* 
   * Bakery Algorithm 
   */
  static class Bakery { 
    private int n; 
    private AtomicBoolean [] choosing; 
    private AtomicInteger [] number; 

    public Bakery(int n_that) {
      this.n        = n_that; 
      this.choosing = new AtomicBoolean[n]; 
      this.number   = new AtomicInteger[n]; 
      for (int i = 0; i < n; i++) {
        choosing[i] = new AtomicBoolean(); 
          number[i] = new AtomicInteger(); 
      }
    }

    public void lock(int i) {
      choosing[i].set(true); 
      int num_i = number[0].get(); 
      for (int j = 0; j < n; j++) {
        int num_j = number[j].get(); 
        if (num_j > num_i) 
          num_i = num_j; 
      }
      number  [i].set(++num_i); 
      choosing[i].set(false); 

      for (int j = 0; j < n; j++) {
        while (choosing[j].get()); 
        while ((number[j].get() != 0) && 
              ((number[j].get() <  num_i) || 
              ((number[j].get() == num_i) && j < i)));
      }
      // System.out.println("PID " + i + " enters CS"); 
    }

    public void unlock(int i) {
      number[i].set(0); 
      // System.out.println("PID " + i + " leaves CS"); 
    }
  }

  static class BakeryInc implements Runnable { 
      private static Bakery lock; 
      public static volatile int c = 0; 
      public int m; 
      public int n; 
      public int pid; 

      public BakeryInc(int m_that, int n_that, int pid_that) { 
          this.m    = m_that; 
          this.n    = n_that; 
          this.pid  = pid_that; 
      }

      public static void newLock(int n_val) {
        BakeryInc.lock = new Bakery(n_val); 
      }

      public void run() { 
        for ( int i = 0; i < Math.ceil(m/(double)n); i++ ) {
          lock.lock(pid); 
          try {
            c++; 
          } finally {
            lock.unlock(pid); 
          }
        }
      }

      public static void doit(int n_thread) { 
          int m = 1200000; 
          int n = n_thread; 
  
          BakeryInc [] f_array = new BakeryInc[n]; 
          Thread    [] t_array = new Thread[n]; 

          BakeryInc.newLock(n); 
  
          for (int i = 0; i < n; i++ ) { 
              f_array[i] = new BakeryInc(m, n, i); 
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
          long exeTime = endTime - startTime; 
          System.out.println("With " + n + " threads: " + exeTime + " ms / final counter value: " + c); 
      }
  }

  public static void main(String[] args) {
    int [] numThreads = new int[]{1, 2, 4, 8}; 

    /* (a) */ 
    System.out.println( "#-------------------------------------------------"); 
    System.out.println( "#> (a) Using Lamport's Fast Mutex"); 
    System.out.println( "#-------------------------------------------------");

    for (int i : numThreads) {
      FastMutexInc.c = 0; 
      FastMutexInc.doit(i); 
    }

    /* (b) */ 
    System.out.println( "#-------------------------------------------------"); 
    System.out.println( "#> (b) Using Bakery Algorithm"); 
    System.out.println( "#-------------------------------------------------");

    for (int i : numThreads) {
      BakeryInc.c = 0; 
      BakeryInc.doit(i); 
    }

  }
      
}
