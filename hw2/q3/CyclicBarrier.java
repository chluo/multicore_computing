/**
 * Created by wenwen on 9/23/16.
 */
import java.util.concurrent.Semaphore;

public class CyclicBarrier {
    int parties;
    volatile int partiesWaiting;
    Semaphore mutex;
    Semaphore gate;
    public CyclicBarrier(int parties){
        this.parties = parties;
        partiesWaiting = parties;
        gate = new Semaphore(0,true);
        mutex = new Semaphore(1,true);
    }
    int await() throws InterruptedException{
        mutex.acquire();
        partiesWaiting--;
        int arrivalIndex = partiesWaiting;
        mutex.release();
        if(partiesWaiting!=0) gate.acquire();

        if(arrivalIndex == 0) {
            gate.release(parties - 1);
            partiesWaiting = parties;
        }
        return arrivalIndex;
    }
}
