package q2;

import java.lang.* ; 
import java.util.concurrent.atomic.* ; 
import java.util.concurrent.locks.*  ;

// TODO
// Use MyLock to protect the count

public class LockCounter extends Counter {
    private MyLock myLock; 
    public LockCounter(MyLock lock) {
      this.count  = 0; 
      this.myLock = lock; 
    }

    @Override
    public void increment(int myId) {
      myLock.lock(myId); 
      try {
        count ++; 
      } finally {
        myLock.unlock(myId); 
      }
    }

    @Override
    public int getCount() {
        return count;
    }
}
