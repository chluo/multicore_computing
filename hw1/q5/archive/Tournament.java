import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Created by wenwen on 9/9/16.
 */
public class Tournament {

    int numThreads;
    PertersonAlgorithm[] entryGates;
    public static int defaultSum = 1200000;

    public Tournament(int numThreads){
        this.numThreads = numThreads;
        entryGates = new PertersonAlgorithm[numThreads/2];
        for(int i = 0; i < numThreads/2; i++){
            entryGates[i] = new PertersonAlgorithm(i);
        }
        buildTournament(entryGates);

    }

    public void buildTournament(PertersonAlgorithm[] entryGates){
        if(entryGates.length==0) return;
        int num = entryGates.length/2;
        if(num==0) {
            entryGates[0].nextGate = null;
            return;
        }
        PertersonAlgorithm[] nextGates = new PertersonAlgorithm[num];
        for(int i = 0; i < num; i++){
            nextGates[i] = new PertersonAlgorithm(i);
            entryGates[i*2].nextGate = nextGates[i];
            entryGates[i*2+1].nextGate = nextGates[i];
        }
        buildTournament(nextGates);
    }

    public LinkedList<PertersonAlgorithm> requestCS(int pid){
        LinkedList<PertersonAlgorithm> path= new LinkedList<>();
        if(entryGates.length==0) return path;
        PertersonAlgorithm entryGate = entryGates[pid/2];
        entryGate.lock(pid%2);
        entryGate.lockedNum = pid%2;
        path.add(entryGate);
        while(entryGate.nextGate!=null){
            pid = entryGate.standard%2;
            entryGate = entryGate.nextGate;
            entryGate.lock(pid);
            entryGate.lockedNum = pid;
            path.add(entryGate);
        }
        // reach the last gate and then enter into CS
        return path;
    }

    public void releaseCS(LinkedList<PertersonAlgorithm> path){
        if(path.isEmpty()) return;
        while (!path.isEmpty()) {
            PertersonAlgorithm gate = path.removeLast();
            gate.unlock(gate.lockedNum);
        }
    }

    static class PertersonAlgorithm {
        PertersonAlgorithm nextGate;
        int lockedNum;
        int standard;
        volatile boolean wantCS[] = {false, false};
        volatile  int turn = 1;
        public PertersonAlgorithm(int i){
            standard = i;
        }
        public void lock(int i) {
            int j = 1 - i;
            wantCS[i] = true;
            turn = j;
            while (wantCS[j] && (turn == j)) ;
        }
        public void unlock(int i) {
            wantCS[i] = false;
        }
    }

    static class Incrementer implements Runnable{
        public static volatile int c = 0;
        public int pid;
        public static Tournament tournamentLock;
        int target;

        public Incrementer(int pid, int numThreads){
            this.pid = pid;
            tournamentLock = new Tournament(numThreads);
            target = (int)Math.ceil(defaultSum/numThreads);
        }

        public void run(){
            for(int i = 0; i < target; i++) {
                LinkedList<PertersonAlgorithm> path = tournamentLock.requestCS(pid);
                c++;
                tournamentLock.releaseCS(path);
            }
        }
    }

    public static void main(String[] args) throws InterruptedException {
        int[] numThreads = new int[]{1,2,4,8};

        for(int numThread: numThreads){
            Thread[] threads = new Thread[numThread];
            for(int j = 0; j < numThread; j++){
                threads[j] = new Thread(new Incrementer(j,numThread));
            }
            long startTime = System.currentTimeMillis();
            for(int j = 0; j< numThread; j++){
                threads[j].start();
            }
            for (int j = 0; j< numThread; j++){
                threads[j].join();
            }
            long endTime = System.currentTimeMillis();
            long time  = endTime - startTime;

            System.out.println("With "+ numThread +" threads: " + time + " ms.");
        }
    }
}
