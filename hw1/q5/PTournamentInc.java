import java.lang.* ; 
import java.util.concurrent.locks.ReentrantLock; 

public class PTournamentInc implements Runnable { 
    public boolean complete = true; 
    public static volatile int c = 0; 
    public int m; 
    public int n; 
    private static PTournamentLock lock; 
    public PTournamentInc(int m_val, int n_val) { 
        m = m_val; 
        n = n_val; 
        lock = new PTournamentLock(n); 
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

        /*
        PTournamentInc t; 
        for ( int i = 0; i < n; i++ ) {
            t = new PTournamentInc(m, n); 
            t.run(); 
        }
        */

        PTournamentInc t1  = new PTournamentInc(m, n);  
        PTournamentInc t2  = new PTournamentInc(m, n);  
        PTournamentInc t3  = new PTournamentInc(m, n);  
        PTournamentInc t4  = new PTournamentInc(m, n);  
        PTournamentInc t5  = new PTournamentInc(m, n);  
        PTournamentInc t6  = new PTournamentInc(m, n);  
        PTournamentInc t7  = new PTournamentInc(m, n);  
        PTournamentInc t8  = new PTournamentInc(m, n);  

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
