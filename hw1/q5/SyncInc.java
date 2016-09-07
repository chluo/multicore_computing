import java.lang.* ; 

public class SyncInc implements Runnable { 
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
    public static void main(String[] args) { 
        int m = 1200000; 
        int n = Integer.parseInt(args[0]); 

        SyncInc t1 = new SyncInc(m, n); 
        SyncInc t2 = new SyncInc(m, n); 
        SyncInc t3 = new SyncInc(m, n); 
        SyncInc t4 = new SyncInc(m, n); 
        SyncInc t5 = new SyncInc(m, n); 
        SyncInc t6 = new SyncInc(m, n); 
        SyncInc t7 = new SyncInc(m, n); 
        SyncInc t8 = new SyncInc(m, n); 

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
