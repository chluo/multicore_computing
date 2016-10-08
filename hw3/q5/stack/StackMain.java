package stack;

import java.util.ArrayList;

public class StackMain {
	public static LockFreeStack q = new LockFreeStack(); 
	public static ArrayList<Integer> res = new ArrayList<>(); 
	
	public static class thread1 implements Runnable {
		public void run() {
			q.push(1); 
			q.push(2); 
			try {
				res.add(q.pop());
			} catch (EmptyStack e) {
				e.printStackTrace();
			}
			try {
				res.add(q.pop());
			} catch (EmptyStack e) {
				e.printStackTrace();
			}
		}
	}
	
	public static class thread2 implements Runnable {
		public void run() {
			q.push(3); 
			q.push(4); 
			try {
				res.add(q.pop());
			} catch (EmptyStack e) {
				e.printStackTrace();
			} 
			try {
				res.add(q.pop());
			} catch (EmptyStack e) {
				e.printStackTrace();
			}
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