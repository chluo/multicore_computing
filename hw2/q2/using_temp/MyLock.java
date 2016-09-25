package q2;

import java.lang.* ; 
import java.util.concurrent.atomic.* ; 
import java.util.concurrent.locks.*  ;

public interface MyLock{
    public void lock(int myId);
    public void unlock(int myId);
}
