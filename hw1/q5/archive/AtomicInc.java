import java.lang.* ; 
import java.util.concurrent.atomic.* ; 

public class AtomicInc implements Runnable { 
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
    public static void main(String[] args) { 
        int m = 1200000; 
        int n = Integer.parseInt(args[0]); 

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
