package queue;

import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.ReentrantLock; 

public class LockQueue implements MyQueue {
	// you are free to add members
	public AtomicInteger count; 
	private ReentrantLock enqLock; 
	private ReentrantLock deqLock; 
	private Node head; 
	private Node tail; 
	private Node dummy; 
	public LockQueue() {
		// implement your constructor here
		this.count = new AtomicInteger(0); 
		this.enqLock = new ReentrantLock(); 
		this.deqLock = new ReentrantLock();  
		this.dummy = new Node(-1); 
		this.head = dummy; 
		this.tail = dummy; 
	}
	public boolean enq(Integer value) {
		// implement your enq method here
		enqLock.lock();
		try {
			Node newNode = new Node(value); 
			tail.next = newNode; 
			tail = newNode; 
			count.getAndIncrement(); 
			return true; 
		} catch (Exception e) {
			return false; 
		} finally {
			enqLock.unlock();
		}
	}
	public Integer deq() {
		// implement your deq method here
		int res; 
		deqLock.lock(); 
		try {
			while (count.get() == 0);
			res = head.next.value; 
			head = head.next; 
			return res; 
		} catch (Exception e) {
			return null; 
		} finally {
			deqLock.unlock();
		}
	}
	protected class Node {
		public Integer value;
		public Node next;
		public Node(Integer x) {
			value = x;
			next = null;
		}
	}
}
