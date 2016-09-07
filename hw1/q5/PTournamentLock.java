import java.lang.* ; 

public class PTournamentLock { 
    /* Static counter counting PIDs */
    private static int pid = 0; 

    /* Number of threads */
    private int n; 

    /* The gate number each thread is at: gate[thread#] */
    /* Gates and levels are numbered from root to leafs */
    private int [] gate; 
    /* The stuck thread at each gate: last[gate#] */
    private volatile int [] last; 

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

    /* Get the level of a certain gate */ 
    /* 
    private int getLevel(int gate) { 
        return Math.log10(gate) / Math.log10(2.0) + 1; 
    }
    */

    /* CTOR */
    public PTournamentLock(int n_val) {
        n = n_val; 
        gate  = new int[n + 1]; 
        last  = new int[getNumGate() + 1]; 
        /*
        System.out.println("Number of gates: " + getNumGate()); 
        */
    }

    /* The i-th thread */
    /* i = 1: n */
    /* k = 1: gate.length */
    public void lock() {
        int i = ++pid; 
        int k = last.length - (i + 1)/2; 
        while ( k > 0 ) {
            gate[i] = k; 
            last[k] = i; 

            /* Check if there is anyone ahead in the thread's path */
            boolean someone_ahead = true; 
            while ( someone_ahead && last[k] == i ) {
                someone_ahead = false;
                for ( int j = 1; j < n + 1; j++ ) {
                    int check_gate = k;  
                    while ( check_gate > 1 ) {
                        check_gate = check_gate/2; 
                        if ( j != i && gate[j] == check_gate )  {
                            someone_ahead = true;
                            break; 
                        }

                    }
                }
            }

            k = k/2; 
        }
    }

    public void unlock() {
        gate[pid] = 0; 
    }
}
