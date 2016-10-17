package q3;

import java.util.concurrent.locks.*; 

public class Garden {
	// you are free to add members
	private static int MAX; 
	private static int numEmptyH = 0; 
	private static int numSeeded = 0;
	private static boolean shovelAvail = true; 
	private static ReentrantLock gardenLock = new ReentrantLock(); 
	private static Condition condDigging = gardenLock.newCondition(); 
	private static Condition condSeeding = gardenLock.newCondition(); 
	private static Condition condFilling = gardenLock.newCondition(); 

	public Garden(int max){
		// implement your constructor here
		MAX = max; 
	}
	public void startDigging() throws InterruptedException{
        // implement your startDigging method here
		gardenLock.lock(); 
		try {
			while ((numEmptyH > MAX) || !shovelAvail) 
				condDigging.await();
			shovelAvail = false;
			System.out.println("[Newton] starts digging a hole ..."); 
			System.out.println("[Newton] number of empty  holes: " + numEmptyH); 
			System.out.println("[Newton] number of seeded holes: " + numSeeded);
		} finally {
			gardenLock.unlock(); 
		}
	}
	public void doneDigging(){
        // implement your doneDigging method here
		gardenLock.lock(); 
		try {
			numEmptyH ++; 
			shovelAvail = true; 
			condSeeding.signal();
			condFilling.signal();
			System.out.println("[Newton] finishes digging the hole ...");
			System.out.println("[Newton] number of empty  holes: " + numEmptyH); 
			System.out.println("[Newton] number of seeded holes: " + numSeeded); 
		} finally {
			gardenLock.unlock();
		}
	} 
	public void startSeeding() throws InterruptedException{
        // implement your startSeeding method here
		gardenLock.lock();
		try {
			while (!(numEmptyH > 0)) 
				condSeeding.await();
			System.out.println("[Ben   ] starts seeding a hole ...");
			System.out.println("[Ben   ] number of empty  holes: " + numEmptyH); 
			System.out.println("[Ben   ] number of seeded holes: " + numSeeded);
		} finally {
			gardenLock.unlock();
		}
	}
	public void doneSeeding(){
        // implement your doneSeeding method here
		gardenLock.lock();
		try { 
			numEmptyH --; 
			numSeeded ++; 
			condFilling.signal();
			condDigging.signal();		
			System.out.println("[Ben   ] finishes seeding the hole ...");
			System.out.println("[Ben   ] number of empty  holes: " + numEmptyH); 
			System.out.println("[Ben   ] number of seeded holes: " + numSeeded);
		} finally {
			gardenLock.unlock();
		}
	} 
	public void startFilling() throws InterruptedException{
        // implement your startFilling method here
		gardenLock.lock();
		try {
			while (!(numSeeded > 0) || !shovelAvail)
				condFilling.await();
			shovelAvail = false; 
			System.out.println("[Mary  ] starts filling a hole ...");
			System.out.println("[Mary  ] number of empty  holes: " + numEmptyH); 
			System.out.println("[Mary  ] number of seeded holes: " + numSeeded);
		} finally {
			gardenLock.unlock();
		}
	}
	public void doneFilling(){
        // implement your doneFilling method here
		gardenLock.lock(); 
		try { 
			numSeeded --; 
			shovelAvail = true; 
			condDigging.signal();
			System.out.println("[Mary  ] finishes filling the hole ...");
			System.out.println("[Mary  ] number of empty  holes: " + numEmptyH); 
			System.out.println("[Mary  ] number of seeded holes: " + numSeeded);
		} finally { 
			gardenLock.unlock();
		}
	}

	// You are free to implements your own Newton, Benjamin and Mary
	// classes. They will NOT count to your grade.
	protected static class Newton implements Runnable {
		Garden garden;
		public Newton(Garden garden){
			this.garden = garden;
		}
		@Override
		public void run() {
		    while (true) {
                try {
					garden.startDigging();
				} catch (InterruptedException e) {}
			    dig();
				garden.doneDigging();
			}
		} 
		
		private void dig(){
			try {
				Thread.sleep(3);
			} catch (InterruptedException e) {}
		}
	}
	
	protected static class Benjamin implements Runnable {
		Garden garden;
		public Benjamin(Garden garden){
			this.garden = garden;
		}
		@Override
		public void run() {
		    while (true) {
                try {
					garden.startSeeding();
				} catch (InterruptedException e) {}
				plantSeed();
				garden.doneSeeding();
			}
		} 
		
		private void plantSeed(){
			try {
				Thread.sleep(1);
			} catch (InterruptedException e) {}
		}
	}
	
	protected static class Mary implements Runnable {
		Garden garden;
		public Mary(Garden garden){
            this.garden = garden;
		}
		@Override
		public void run() {
		    while (true) {
                try {
					garden.startFilling();
				} catch (InterruptedException e) {}
			 	Fill();
			 	garden.doneFilling();
			}
		} 
		
		private void Fill(){
			try {
				Thread.sleep(2);
			} catch (InterruptedException e) {}
		}
	}

}
