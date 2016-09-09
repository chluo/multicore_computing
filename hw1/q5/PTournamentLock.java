import java.lang.* ; 
import java.util.concurrent.locks.* ; 
import java.util.concurrent.atomic.AtomicInteger ; 

public class PTournamentLock extends ReentrantLock { 

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
