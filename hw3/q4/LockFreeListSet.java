package q4;

import java.util.concurrent.atomic.AtomicMarkableReference;

public class LockFreeListSet implements ListSet {
	// you are free to add members
	private Node head; 
	private Node dummy; 
	public LockFreeListSet() {
		// implement your constructor here
		dummy = new Node(-1); 
		head = dummy; 
	}
	public boolean add(int value) {
		// implement your add method here	
		Node pre; 
		Node cur; 
		Node newNode = new Node(value); 
		while (true) {
			pre = head; 
			cur = pre.next.getReference(); 
			while (cur != null && cur.value < value) {
				pre = cur; 
				cur = cur.next.getReference(); 
			}
			if (cur == null || cur.value > value) {
				newNode.next.set(cur, false); 
				if (!pre.next.compareAndSet(cur, newNode, false, false)) 
					continue; 
				return true; 
			}
			return false; 
		}
	}
	public boolean remove(int value) {
		Node pre; 
		Node cur; 
		while (true) {
			pre = head; 
			cur = pre.next.getReference(); 
			while (cur != null && cur.value < value) {
				pre = cur; 
				cur = cur.next.getReference(); 
			}
			if (cur != null && cur.value == value) {
				if (!cur.next.attemptMark(cur.next.getReference(), true)) 
					continue; 
				if (!pre.next.compareAndSet(cur, cur.next.getReference(), false, false))
					continue; 
				return true;  
			}
			return false; 
		}
	}
	public boolean contains(int value) {
		// implement your contains method here	
		Node cur = head.next.getReference(); 
		while (cur != null) {
			if (cur.value == value && !cur.next.isMarked()) 
				return true; 
			cur = cur.next.getReference(); 
		}
		return false;
	}
	protected class Node {
		public Integer value;
		public AtomicMarkableReference<Node> next;
		public Node(Integer x) {
			value = x;
			next = new AtomicMarkableReference<>(null, false); 
		}
	}
/*
	// you are free to add members
	private AtomicMarkableReference<Node> head; 
	private Node dummy; 
	public LockFreeListSet() {
		// implement your constructor here
		dummy = new Node(-1); 
		head = new AtomicMarkableReference<>(dummy, false); 
	}
	public boolean add(int value) {
		// implement your add method here	
		AtomicMarkableReference<Node> pre = new AtomicMarkableReference<>(null, false); 
		AtomicMarkableReference<Node> cur = new AtomicMarkableReference<>(null, false); 
		Node newNode = new Node(value); 
		while (true) {
			pre.set(head.getReference(), head.isMarked());
			cur.set(pre.getReference().next.getReference(), pre.getReference().next.isMarked());
			while (cur.getReference() != null && cur.getReference().value < value) {
				pre.set(cur.getReference(), cur.isMarked());
				cur.set(cur.getReference().next.getReference(), cur.getReference().next.isMarked()); 
			}
			if (cur.getReference() == null || cur.getReference().value > value) {
				newNode.next.set(cur.getReference(), cur.isMarked()); 
				if (pre.isMarked()) 
					continue; 
				if (!pre.getReference().next.compareAndSet(cur.getReference(), newNode, false, false)) 
					continue;  
				return true; 
			}
			return false; 
		}
	}
	public boolean remove(int value) {
		AtomicMarkableReference<Node> pre = new AtomicMarkableReference<>(null, false); 
		AtomicMarkableReference<Node> cur = new AtomicMarkableReference<>(null, false);  
		while (true) {
			pre.set(head.getReference(), head.isMarked());
			cur.set(pre.getReference().next.getReference(), pre.getReference().next.isMarked());
			while (cur.getReference() != null && cur.getReference().value < value) {
				pre.set(cur.getReference(), cur.isMarked());
				cur.set(cur.getReference().next.getReference(), cur.getReference().next.isMarked()); 
			}
			if (cur.getReference() != null && cur.getReference().value == value) {
				if (pre.isMarked()) 
					continue; 
				if (!pre.getReference().next.attemptMark(cur.getReference(), true)) 
					continue; 
				if (!pre.getReference().next.compareAndSet(cur.getReference(), cur.getReference().next.getReference(), true, cur.getReference().next.isMarked())) 
					continue; 
				return true;  
			}
			return false; 
		}
	}
	public boolean contains(int value) {
		// implement your contains method here	
		AtomicMarkableReference<Node> cur = new AtomicMarkableReference<>(head.getReference().next.getReference(), head.getReference().next.isMarked());
		while (cur.getReference() != null) {
			if (cur.getReference().value == value && !cur.isMarked()) 
				return true; 
			cur.set(cur.getReference().next.getReference(), cur.getReference().next.isMarked());
		}
		return false;
	}
	protected class Node {
		public Integer value;
		public AtomicMarkableReference<Node> next;
		public Node(Integer x) {
			value = x;
			next = new AtomicMarkableReference<>(null, false); 
		}
	}
*/
}
