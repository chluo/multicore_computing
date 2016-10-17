package q4;

public class ListMain {
	public static class CoarseGrainedListTest {
		public static CoarseGrainedListSet q = new CoarseGrainedListSet(); 
		public static class thread1 implements Runnable {
			public void run() { 
				q.add(1);
				q.add(2); 
				while (!q.contains(3)); 
				q.remove(3); 
			}
		}
		public static class thread2 implements Runnable {
			public void run() {
				q.add(3); 
				q.add(4); 
				while (!q.contains(5)); 
				q.remove(5); 
			}
		}
		public static class thread3 implements Runnable {
			public void run() { 
				q.add(5);  
				q.add(6); 
				while (!q.contains(7)); 
				q.remove(7); 
			}
		}
		public static class thread4 implements Runnable {
			public void run() {
				q.add(7); 
				q.add(8); 
				while (!q.contains(1)); 
				q.remove(1); 
			}
		}
		public static void printRes () {
			System.out.println("CoarseGrained: "); 
			for (int i = 1; i < 9; i++) 
				System.out.print(q.contains(i) + " ");
			System.out.print("\n");
		}
	}
	public static class FineGrainedListTest {
		public static FineGrainedListSet q = new FineGrainedListSet(); 
		public static class thread1 implements Runnable {
			public void run() { 
				q.add(1);
				q.add(2); 
				while (!q.contains(3)); 
				q.remove(3); 
			}
		}
		public static class thread2 implements Runnable {
			public void run() {
				q.add(3); 
				q.add(4); 
				while (!q.contains(5)); 
				q.remove(5); 
			}
		}
		public static class thread3 implements Runnable {
			public void run() { 
				q.add(5);  
				q.add(6); 
				while (!q.contains(7)); 
				q.remove(7); 
			}
		}
		public static class thread4 implements Runnable {
			public void run() {
				q.add(7); 
				q.add(8); 
				while (!q.contains(1)); 
				q.remove(1); 
			}
		}
		public static void printRes () {
			System.out.println("FineGrained: "); 
			for (int i = 1; i < 9; i++) 
				System.out.print(q.contains(i) + " ");
			System.out.print("\n");
		}
	}
	public static class LockFreeListTest {
		public static LockFreeListSet q = new LockFreeListSet(); 
		public static class thread1 implements Runnable {
			public void run() { 
				q.add(1);
				q.add(2); 
				while (!q.contains(3)); 
				q.remove(3); 
			}
		}
		public static class thread2 implements Runnable {
			public void run() {
				q.add(3); 
				q.add(4); 
				while (!q.contains(5)); 
				q.remove(5); 
			}
		}
		public static class thread3 implements Runnable {
			public void run() { 
				q.add(5);  
				q.add(6); 
				while (!q.contains(7)); 
				q.remove(7); 
			}
		}
		public static class thread4 implements Runnable {
			public void run() {
				q.add(7); 
				q.add(8); 
				while (!q.contains(1)); 
				q.remove(1); 
			}
		}
		public static void printRes () {
			System.out.println("LockFree: "); 
			for (int i = 1; i < 9; i++) 
				System.out.print(q.contains(i) + " ");
			System.out.print("\n");
		}
	}
	public static void main(String args[]) {
		{
			CoarseGrainedListTest.thread1 t1 = new CoarseGrainedListTest.thread1(); 
			CoarseGrainedListTest.thread2 t2 = new CoarseGrainedListTest.thread2(); 
			CoarseGrainedListTest.thread3 t3 = new CoarseGrainedListTest.thread3(); 
			CoarseGrainedListTest.thread4 t4 = new CoarseGrainedListTest.thread4(); 
			Thread th1 = new Thread(t1); 
			Thread th2 = new Thread(t2);
			Thread th3 = new Thread(t3); 
			Thread th4 = new Thread(t4);

			th1.start();
			th2.start();
			th3.start();
			th4.start();
			try {
				th1.join();
				th2.join();
				th3.join();
				th4.join(); 
			} catch (Exception e) {}

			CoarseGrainedListTest.printRes(); 
		}
		{
			FineGrainedListTest.thread1 t1 = new FineGrainedListTest.thread1(); 
			FineGrainedListTest.thread2 t2 = new FineGrainedListTest.thread2(); 
			FineGrainedListTest.thread3 t3 = new FineGrainedListTest.thread3(); 
			FineGrainedListTest.thread4 t4 = new FineGrainedListTest.thread4(); 
			Thread th1 = new Thread(t1); 
			Thread th2 = new Thread(t2);
			Thread th3 = new Thread(t3); 
			Thread th4 = new Thread(t4);

			th1.start();
			th2.start();
			th3.start();
			th4.start();
			try {
				th1.join();
				th2.join();
				th3.join();
				th4.join(); 
			} catch (Exception e) {}

			FineGrainedListTest.printRes(); 
		}
		{
			LockFreeListTest.thread1 t1 = new LockFreeListTest.thread1(); 
			LockFreeListTest.thread2 t2 = new LockFreeListTest.thread2(); 
			LockFreeListTest.thread3 t3 = new LockFreeListTest.thread3(); 
			LockFreeListTest.thread4 t4 = new LockFreeListTest.thread4(); 
			Thread th1 = new Thread(t1); 
			Thread th2 = new Thread(t2);
			Thread th3 = new Thread(t3); 
			Thread th4 = new Thread(t4);

			th1.start();
			th2.start();
			th3.start();
			th4.start();
			try {
				th1.join();
				th2.join();
				th3.join();
				th4.join(); 
			} catch (Exception e) {}

			LockFreeListTest.printRes(); 
		}
	}
}