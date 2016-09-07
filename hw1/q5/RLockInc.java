import java.lang.* ; 
import java.util.concurrent.locks.ReentrantLock; 

public class RLockInc implements Runnable { 
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
    public static void main(String[] args) { 
        int m = 1200000; 
        int n = Integer.parseInt(args[0]); 

        RLockInc t1 = new RLockInc(m, n); 
        RLockInc t2 = new RLockInc(m, n); 
        RLockInc t3 = new RLockInc(m, n); 
        RLockInc t4 = new RLockInc(m, n); 
        RLockInc t5 = new RLockInc(m, n); 
        RLockInc t6 = new RLockInc(m, n); 
        RLockInc t7 = new RLockInc(m, n); 
        RLockInc t8 = new RLockInc(m, n); 

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
