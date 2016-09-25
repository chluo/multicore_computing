package q2;

import java.lang.* ; 
import java.util.concurrent.atomic.* ; 
import java.util.concurrent.locks.*  ;

// TODO
// Implement the bakery algorithm

public class BakeryLock implements MyLock {
    private int n; 
    private AtomicBoolean [] choosing; 
    private AtomicInteger [] number; 

    public BakeryLock(int numThread) {
      this.n = numThread; 
      this.choosing = new AtomicBoolean[n]; 
      this.number   = new AtomicInteger[n]; 
      for (int i = 0; i < n; i++) {
        choosing[i] = new AtomicBoolean(); 
          number[i] = new AtomicInteger(); 
      }
    }

    @Override
    public void lock(int myId) {
      choosing[myId].set(true); 
      int num_i = number[0].get(); 
      for (int j = 0; j < n; j++) {
        int num_j = number[j].get(); 
        if (num_j > num_i) 
          num_i = num_j; 
      }
      number  [myId].set(++num_i); 
      choosing[myId].set(false); 

      for (int j = 0; j < n; j++) {
        while (choosing[j].get()); 
        while ((number[j].get() != 0) && 
              ((number[j].get() <  num_i) || 
              ((number[j].get() == num_i) && j < myId)));
      }
    }

    @Override
    public void unlock(int myId) {
      number[myId].set(0); 
    }
}
