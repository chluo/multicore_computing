package stack;

import java.util.concurrent.locks.*; 

public class LockStack implements MyStack {
	// you are free to add members
	private Node head;
	private ReentrantLock stackLock; 	
	
  public LockStack() {
	  // implement your constructor here
	  this.stackLock = new ReentrantLock(); 
  }
  
  public boolean push(Integer value) {
	  // implement your push method here
	  stackLock.lock();
	  try {
		  Node newNode = new Node(value);
		  if (head != null) 
			  newNode.next = head;
		  head = newNode; 
		  return true; 
	  } catch (Exception e) {
		  return false; 
	  } finally {
		  stackLock.unlock();
	  }
  }
  
  public Integer pop() throws EmptyStack {
	  // implement your pop method here
	  stackLock.lock();
	  try {
		  if (head == null) throw new EmptyStack(); 
		  Node popNode = head; 
		  head = head.next; 
		  popNode.next = null; 
		  return popNode.value; 
	  } catch (Exception e) {
		  return null; 
	  } finally { 
		  stackLock.unlock();
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
