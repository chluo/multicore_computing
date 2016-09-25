package q2;

import java.lang.* ; 
import java.util.concurrent.atomic.* ; 
import java.util.concurrent.locks.*  ;

// TODO 
// Implement Fast Mutex Algorithm
public class FastMutexLock implements MyLock {
    private int n; 
    private volatile int x = -1; 
    private volatile int y = -1; 
    private AtomicInteger [] flag; 

    public FastMutexLock(int numThread) {
      this.n    = numThread; 
      this.flag = new AtomicInteger[n]; 
      for (int i = 0; i < n; i++) {
        flag[i] = new AtomicInteger(0); 
      }
    }

    @Override
    public void lock(int myId) {
      while(true) { 
        flag[myId].set(1); 
        x = myId; 
        if (y != -1) {
          flag[myId].set(0); 
          while (y != -1); 
          continue; 
        }
        else {
          y = myId; 
          if (x == myId) {
            return; 
          }
          else {
            flag[myId].set(0); 
            for (int j = 0; j < flag.length; j++) 
              while (flag[j].get() == 1); 
            if (y == myId) {
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

    @Override
    public void unlock(int myId) {
      y = -1; 
      flag[myId].set(0); 
    }
}
