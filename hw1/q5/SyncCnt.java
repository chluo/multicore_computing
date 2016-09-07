import java.lang.* ; 

public class SyncCnt {
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
