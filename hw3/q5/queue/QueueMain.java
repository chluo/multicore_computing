package queue;

import java.util.ArrayList;

public class QueueMain {
	public static LockQueue q = new LockQueue(); 
	public static ArrayList<Integer> res = new ArrayList<>(); 
	
	public static class thread1 implements Runnable {
		public void run() {
			// System.out.println("thread 1"); 
			q.enq(1); 
			q.enq(2); 
			res.add(q.deq());
			res.add(q.deq());
		}
	}
	
	public static class thread2 implements Runnable {
		public void run() {
			// System.out.println("thread 2");
			q.enq(3); 
			q.enq(4); 
			res.add(q.deq()); 
			res.add(q.deq());
		}
	}
	
	public static void printRes () {
		for (int i = 0; i < res.size(); i++) 
			System.out.print(res.get(i) + " ");
		System.out.print("\n");
	}
	
	public static void main(String args[]) {
		thread1 t1 = new thread1(); 
		thread2 t2 = new thread2(); 
		Thread th1 = new Thread(t1); 
		Thread th2 = new Thread(t2); 
		
		th1.start();
		th2.start();
		try {
			th1.join();
			th2.join();
		} catch (Exception e) {}
		
		printRes(); 
	}
}