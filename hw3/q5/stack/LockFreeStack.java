package stack;

import java.util.concurrent.atomic.*;

public class LockFreeStack implements MyStack {
	// you are free to add members
	AtomicReference<Node> head; 	
	public LockFreeStack() {
		// implement your constructor here
		this.head = new AtomicReference<>(); 
	}	
	public boolean push(Integer value) {
		// implement your push method here
		Node curHead;  
		Node newNode = new Node(value); 	   
		while (true) {
			curHead = head.get(); 
			newNode.next = curHead; 
			if (head.compareAndSet(curHead, newNode)) break;
			// Back off in case of failure
			else Thread.yield(); 
		}	  
		return true;
	} 
	public Integer pop() throws EmptyStack {
		// implement your pop method here
		Node curHead; 
		Node curNext; 
		Integer  res; 	  
		while (true) {
			curHead = head.get();
			curNext = head.get().next;
			if (curHead == null) {
				throw new EmptyStack(); 
			}
			res = curHead.value;
			if (head.compareAndSet(curHead, curNext)) break; 
			// Back off in case of failure
			else Thread.yield();
		}
		return res;
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
