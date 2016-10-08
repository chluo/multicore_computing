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
	  ThreadLocal<Node> curHead = new ThreadLocal<>();  
	  ThreadLocal<Node> newNode = new ThreadLocal<>(); 
	  newNode.set(new Node(value));
	   
	  while (true) {
		  curHead.set(head.get());
		  if (curHead.get() == head.get()) { 
			  if (curHead.get() != null) 
				  newNode.get().next = curHead.get(); 
			  if (head.compareAndSet(curHead.get(), newNode.get())) 
				  break;
		  }
		  // Back off in case of failure
		  try {
			Thread.sleep(1);
		  } catch (InterruptedException e) {}
	  }
	  
	  return true;
  }
  
  public Integer pop() throws EmptyStack {
	  // implement your pop method here
	  ThreadLocal<Node> curHead = new ThreadLocal<>(); 
	  ThreadLocal<Node> curNext = new ThreadLocal<>(); 
	  ThreadLocal<Integer>  res = new ThreadLocal<>(); 
	  
	  while (true) {
		  curHead.set(head.get());
		  curNext.set(head.get().next);
		  if (curHead.get() == head.get()) {
			  if (curHead.get() == null) {
				  throw new EmptyStack(); 
			  }
			  res.set(curHead.get().value);
			  if (head.compareAndSet(curHead.get(), curNext.get())) 
				  break; 
		  }
		  // Back off in case of failure
		  try {
			Thread.sleep(1);
		  } catch (InterruptedException e) {}
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
