package queue;

import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.*; 

public class LockQueue implements MyQueue {
	// you are free to add members
	public AtomicInteger count; 
	// private ReentrantLock enqLock; 
	// private ReentrantLock deqLock; 
	// private Condition notEmpty; 
	private Semaphore enqLock; 
	private Semaphore deqLock; 
	private Semaphore notEmpty; 
	private Node head; 
	private Node tail; 
	private Node dummy; 

  public LockQueue() {
	// implement your constructor here
	this.count = new AtomicInteger(0); 
	// this.enqLock = new ReentrantLock(); 
	// this.deqLock = new ReentrantLock(); 
	// this.notEmpty = deqLock.newCondition(); 
	this.enqLock = new Semaphore(1);
	this.deqLock = new Semaphore(1); 
	this.notEmpty = new Semaphore(1); 
	this.dummy = new Node(-1); 
	this.head = dummy; 
	this.tail = dummy; 
  }
  
  public boolean enq(Integer value) {
	// implement your enq method here
	// enqLock.lock();
	// System.out.println("Try enq: " + value); 
	try {
		enqLock.acquire(); 
		Node newNode = new Node(value); 
		tail.next = newNode; 
		tail = newNode; 
		count.getAndIncrement(); 
		if (notEmpty.availablePermits() == 0) {
			notEmpty.release();
		}
		return true; 
	} catch (Exception e) {
		// e.printStackTrace();
		return false; 
	} finally {
		// enqLock.unlock();
		enqLock.release();
	}
  }
  
  public Integer deq() {
	// implement your deq method here
	// deqLock.lock();
    // System.out.println("Try deq: "); 
	try {
		deqLock.acquire();
		while (count.get() == 0) {
			// notEmpty.await();
			deqLock.release();
			notEmpty.acquire();
			deqLock.acquire();
		}
		ThreadLocal<Node> deqNode = new ThreadLocal<>(); 
		deqNode.set(head.next); 
		head.next = deqNode.get().next;  
		count.getAndDecrement(); 
		return deqNode.get().value; 
	} catch (Exception e) {
		e.printStackTrace();
		return null; 
	} finally {
		// deqLock.unlock();
		deqLock.release();
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
