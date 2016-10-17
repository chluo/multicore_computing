package q4;

import java.util.concurrent.locks.ReentrantLock;

public class FineGrainedListSet implements ListSet {
	// you are free to add members
	private Node dummy; 
	private Node head; 
	public FineGrainedListSet() {
		// implement your constructor here
		dummy = new Node(-1); 
		head = dummy; 
	}
	public boolean add(int value) {
		// implement your add method here	
		Node newNode = new Node(value);
		while (true) { // in case of retry
			Node pre = head; 
			pre.lock.lock();
			Node cur = pre.next;
			if (cur != null) 
				cur.lock.lock();
			try {
				while (cur != null && cur.value < value) {
					pre.lock.unlock();
					pre = cur; 
					cur = cur.next; 
					if (cur != null)
						cur.lock.lock(); 
				}
				if (cur == null || cur.value > value) {
					if (!pre.isDeleted && (cur == null || !cur.isDeleted) && pre.next == cur) {
						newNode.next = cur; 
						pre.next = newNode; 
						return true; 
					}
					continue; 
				}
				return false; 
			} catch (Exception e) {e.printStackTrace();} finally {
				pre.lock.unlock();
				if (cur != null) 
					cur.lock.unlock();
			}
		}
	}
	public boolean remove(int value) {
		// implement your remove method here
		while (true) { // in case of retry 
			Node pre = head; 
			pre.lock.lock();
			Node cur = pre.next;
			if (cur != null) 
				cur.lock.lock();
			try {
				while (cur != null && cur.value < value) {
					pre.lock.unlock();
					pre = cur; 
					cur = cur.next; 
					cur.lock.lock(); 
				}
				if (cur != null && cur.value == value) {
					if (!pre.isDeleted && !cur.isDeleted && pre.next == cur) {
						cur.isDeleted = true; 
						pre.next = cur.next; 
						return true; 
					}
					continue; 
				}
				return false; 
			} catch (Exception e) {e.printStackTrace();} finally { 
				pre.lock.unlock();
				if (cur != null) 
					cur.lock.unlock();
			}
		}
	}
	public boolean contains(int value) {
		// implement your contains method here
		Node cur = head.next; 
		while (cur != null) {
			if (cur.value == value && !cur.isDeleted) 
				return true; 
			cur = cur.next; 
		}
		return false;
	}
	protected class Node {
		public Integer value;
		public boolean isDeleted; 
		public ReentrantLock lock; 
		public Node next;

		public Node(Integer x) {
			value = x;
			isDeleted = false; 
			lock = new ReentrantLock(); 
			next = null;
		}
	}
}
