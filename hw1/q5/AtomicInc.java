import java.lang.* ; 
import java.util.concurrent.atomic.* ; 

public class AtomicInc implements Runnable { 
    public boolean complete = true; 
    public static AtomicInteger c = new AtomicInteger(); 
    public static int expect = 0; 
    public int m; 
    public int n; 
    public AtomicInc(int m_val, int n_val) { 
        m = m_val; 
        n = n_val; 
    }
    public void run() { 
        complete = false; 
        for ( int i = 0; i < Math.ceil(m/(double)n); i++ ) {
            c.compareAndSet(expect, expect + 1); 
            expect++; 
        }
        System.out.println("Current counter value: " + c.get()); 
        complete = true; 
    }
    public static void main(String[] args) { 
        int m = 1200000; 
        int n = Integer.parseInt(args[0]); 

        AtomicInc t1 = new AtomicInc(m, n); 
        AtomicInc t2 = new AtomicInc(m, n); 
        AtomicInc t3 = new AtomicInc(m, n); 
        AtomicInc t4 = new AtomicInc(m, n); 
        AtomicInc t5 = new AtomicInc(m, n); 
        AtomicInc t6 = new AtomicInc(m, n); 
        AtomicInc t7 = new AtomicInc(m, n); 
        AtomicInc t8 = new AtomicInc(m, n); 

        long startTime = System.currentTimeMillis(); 

        if (n >= 1) { 
            t1.run(); 
        }
        if (n >= 2) { 
            t2.run(); 
        }
        if (n >= 3) { 
            t3.run(); 
        }
        if (n >= 4) { 
            t4.run(); 
        }
        if (n >= 5) { 
            t5.run(); 
        }
        if (n >= 6) { 
            t6.run(); 
        }
        if (n >= 7) { 
            t7.run(); 
        }
        if (n >= 8) { 
            t8.run(); 
        }
        if (t1.complete && t2.complete && t3.complete && t4.complete && 
            t5.complete && t6.complete && t7.complete && t8.complete    ) {
            long endTime = System.currentTimeMillis(); 
            System.out.println("Execution time: " + (endTime - startTime) + " ms"); 
        }
    }
}
