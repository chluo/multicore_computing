package q2;

import java.lang.* ; 
import java.util.concurrent.atomic.* ; 
import java.util.concurrent.locks.*  ;

public abstract class Counter {
    public Counter() {
        count = 0;
    }
    protected volatile int count;
    public abstract void increment(int myId);
    public abstract int getCount();
}
