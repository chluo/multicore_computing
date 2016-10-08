package queue;

import java.util.concurrent.atomic.*;

public class LockFreeQueue implements MyQueue {
  // you are free to add members
  private AtomicInteger count; 
  private Node dummy; 
  private AtomicReference<Node> head; 
  private AtomicReference<Node> tail; 

  public LockFreeQueue() {
	// implement your constructor here
	this.count = new AtomicInteger(0); 
	this.dummy = new Node(-1); 
	this.head = new AtomicReference<>(); 
	this.tail = new AtomicReference<>(); 
	this.head.set(dummy);
	this.tail.set(dummy);
  }

  public boolean enq(Integer value) {
	// implement your enq method here
	ThreadLocal<Node> newNode = new ThreadLocal<Node>(); 
	ThreadLocal<Node> curTail = new ThreadLocal<Node>(); 
	ThreadLocal<Node> curNext = new ThreadLocal<Node>(); 
	newNode.set(new Node(value)); 
	
	while (true) {
		curTail.set(tail.get()); 
		curNext.set(tail.get().next);
		if (tail.get() == curTail.get()) {
			if (curNext.get() == null) {
				// TODO compareAndSet
				synchronized (this) {
					if (curTail.get().next == curNext.get()) {
						curTail.get().next = newNode.get(); 
						break; 
					}
				}
			}
			else {
				tail.compareAndSet(curTail.get(), curNext.get()); 
			}
		}
	}
	if (tail.compareAndSet(curTail.get(), newNode.get())) {
		count.getAndIncrement(); 
		return true; 
	}
	
    return false;
  }
  
  public Integer deq() {
	// implement your deq method here
	ThreadLocal<Node> curHead = new ThreadLocal<Node>(); 
	ThreadLocal<Node> curTail = new ThreadLocal<Node>(); 
	ThreadLocal<Node> curNext = new ThreadLocal<Node>();
	ThreadLocal<Integer> res = new ThreadLocal<Integer>(); 
	
	while (true) {
		curHead.set(head.get());
		curTail.set(tail.get());
		curNext.set(head.get().next);
		if (curHead.get() == head.get()) {
			if (curHead.get() == curTail.get()) {
				if (curNext.get() == null) 
					continue; 
				else 
					tail.compareAndSet(curTail.get(), curNext.get()); 
			}
			else {
				res.set(curNext.get().value);
				if (head.compareAndSet(curHead.get(), curNext.get())) 
					break; 
			}
		}
	}
    return res.get();
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
