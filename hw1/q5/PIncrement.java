import java.lang.* ; 
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.locks.* ; 
import java.util.concurrent.atomic.* ; 

public class PIncrement {

  /* The incrementing counter based on Peterson's Algorithm */
  static class Tournament {
  
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
  
      public static void main_a(int numThread) {
  
              Thread[] threads = new Thread[numThread];
              for(int j = 0; j < numThread; j++){
                  threads[j] = new Thread(new Incrementer(j,numThread));
              }
              long startTime = System.currentTimeMillis();
              for(int j = 0; j< numThread; j++){
                  threads[j].start();
              }
              for (int j = 0; j< numThread; j++){
                try {
                  threads[j].join();
                } catch (InterruptedException ie) {}
              }
              long endTime = System.currentTimeMillis();
              long time  = endTime - startTime;
  
              System.out.println("With "+ numThread +" threads: " + time + " ms / final counter value: " + Incrementer.c);
      }
  }
  
  /* The incrementing counter based on AtomicInteger */ 
  static class AtomicInc implements Runnable { 
      public static volatile AtomicInteger c = new AtomicInteger(); 
      public int m; 
      public int n; 
      public AtomicInc(int m_val, int n_val) { 
          m = m_val; 
          n = n_val; 
      }
      public void run() { 
          try {
            for ( int i = 0; i < Math.ceil(m/(double)n); i++ ) {
                c.getAndAdd(1); 
            }
          } finally {}
      }
      public static void main_b(int n_thread) { 
          int m = 1200000; 
          int n = n_thread; 
  
          AtomicInc [] f_array; 
          Thread [] t_array; 
  
          f_array = new AtomicInc[n]; 
          t_array = new Thread[n]; 
  
          for (int i = 0; i < n; i++ ) { 
              f_array[i] = new AtomicInc(m, n); 
              t_array[i] = new Thread(f_array[i]); 
          }
  
          long startTime = System.currentTimeMillis(); 
  
          for (int i = 0; i < n; i++ ) { 
              t_array[i].start(); 
          }
  
          for (int i = 0; i < n; i++ ) { 
            try {
              t_array[i].join(); 
            } catch (InterruptedException e) {} 
          }
  
          long endTime = System.currentTimeMillis(); 
          System.out.println("With " + n + " threads: " + (endTime - startTime) + " ms / final counter value: " + c); 
      }
  }

  /* Syncrhonized counter */
  static class SyncCnt {
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
      public void rst() { 
          cnt = 0; 
      }
  }

  /* The incrementing counter based on the synchronized counter */
  static class SyncInc implements Runnable { 
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
          complete = true; 
      }
      public static void main_c(int n_thread) { 
          int m = 1200000; 
          int n = n_thread; 
  
          SyncInc [] f_array; 
          Thread [] t_array; 
  
          f_array = new SyncInc[n]; 
          t_array = new Thread[n]; 
  
          for (int i = 0; i < n; i++ ) { 
              f_array[i] = new SyncInc(m, n); 
              t_array[i] = new Thread(f_array[i]); 
          }
  
          long startTime = System.currentTimeMillis(); 
  
          for (int i = 0; i < n; i++ ) { 
              t_array[i].start(); 
          }
  
          for (int i = 0; i < n; i++ ) { 
            try {
              t_array[i].join(); 
            } catch (InterruptedException e) {} 
          }
  
          long endTime = System.currentTimeMillis(); 
          System.out.println("With " + n + " threads: " + (endTime - startTime) + " ms / final counter value: " + c.get()); 
      }
  }

  /* The incrementing counter based on the Reentrant Lock */
  static class RLockInc implements Runnable { 
      public boolean complete = true; 
      private static ReentrantLock lock = new ReentrantLock(); 
      public static volatile int c = 0; 
      public int m; 
      public int n; 
      public RLockInc(int m_val, int n_val) { 
          m = m_val; 
          n = n_val; 
      }
      public void run() { 
          complete = false; 
          lock.lock(); 
          try { 
              for ( int i = 0; i < Math.ceil(m/(double)n); i++ ) {
                  c++; 
              }
          } finally { 
              lock.unlock(); 
          }
          complete = true; 
      }
      public static void main_d(int n_thread) { 
          int m = 1200000; 
          int n = n_thread; 
  
          RLockInc [] f_array; 
          Thread [] t_array; 
  
          f_array = new RLockInc[n]; 
          t_array = new Thread[n]; 
  
          for (int i = 0; i < n; i++ ) { 
              f_array[i] = new RLockInc(m, n); 
              t_array[i] = new Thread(f_array[i]); 
          }
  
          long startTime = System.currentTimeMillis(); 
  
          for (int i = 0; i < n; i++ ) { 
              t_array[i].start(); 
          }
  
          for (int i = 0; i < n; i++ ) { 
            try {
              t_array[i].join(); 
            } catch (InterruptedException e) {} 
          }
  
          long endTime = System.currentTimeMillis(); 
          System.out.println("With " + n + " threads: " + (endTime - startTime) + " ms / final counter value: " + c); 
      }
  }

  public static void main(String[] args) {
    /* (a) */ 
    System.out.println( "#-------------------------------------------------"); 
    System.out.println( "#> (a) Using Peterson's Tournament Algorithm" );
    System.out.println( "#-------------------------------------------------");

    int [] numThreads = new int[]{1, 2, 4, 8}; 
    for ( int i : numThreads ) {
      Tournament.Incrementer.c = 0; 
      Tournament.main_a(i); 
    }

    /* (b) */ 
    System.out.println( "#-------------------------------------------------"); 
    System.out.println( "#> (b) Using Java's AtomicInteger" );
    System.out.println( "#-------------------------------------------------");

    for ( int i = 1; i < 9; i++ ) {
      AtomicInc.c.getAndSet(0); 
      AtomicInc.main_b(i); 
    }

    /* (c) */ 
    System.out.println( "#-------------------------------------------------"); 
    System.out.println( "#> (c) Using Java's synchronized construct" );
    System.out.println( "#-------------------------------------------------");

    for ( int i = 1; i < 9; i++ ) {
      SyncInc.c.rst(); 
      SyncInc.main_c(i); 
    }

    /* (d) */ 
    System.out.println( "#-------------------------------------------------"); 
    System.out.println( "#> (d) Using Java's ReentrantLock" );
    System.out.println( "#-------------------------------------------------");

    for ( int i = 1; i < 9; i++ ) {
      RLockInc.c = 0; 
      RLockInc.main_d(i); 
    }

  }
      
}
