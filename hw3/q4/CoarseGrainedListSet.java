package q4;

import java.util.concurrent.locks.ReentrantLock;

public class CoarseGrainedListSet implements ListSet {
	// you are free to add members
	private Node dummy; 
	private Node head; 
	private ReentrantLock lock; 
	public CoarseGrainedListSet() {
		// implement your constructor here	
		dummy = new Node(-1); 
		head = dummy; 
		lock = new ReentrantLock(); 
	}
	public boolean add(int value) {
		// implement your add method here
		Node cur; 
		Node pre; 
		Node newNode = new Node(value); 
		lock.lock();
		try {
			pre = head; 
			cur = pre.next; 
			while (cur != null && cur.value < value) {
				pre = cur; 
				cur = cur.next; 
			}
			if (cur == null || cur.value > value) {
				newNode.next = cur; 
				pre.next = newNode; 
				return true; 
			}
			return false;
		} finally {
			lock.unlock(); 
		}
	}
	public boolean remove(int value) {
		// implement your add method here
		Node cur; 
		Node pre; 
		lock.lock();
		try {
			pre = head; 
			cur = pre.next; 
			while (cur != null && cur.value < value) {
				pre = cur; 
				cur = cur.next; 
			}
			if (cur != null && cur.value == value) {
				pre.next = cur.next; 
				return true; 
			}
			return false;
		} finally {
			lock.unlock(); 
		}
	}
	public boolean contains(int value) {
		// implement your contains method here	
		Node cur = head.next; 
		while (cur != null) {
			if (cur.value == value) 
				return true; 
			cur = cur.next; 
		}
		return false;
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
