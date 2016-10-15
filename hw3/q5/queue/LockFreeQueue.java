package queue;

import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.*;

public class LockFreeQueue implements MyQueue {
  // you are free to add members
  private AtomicInteger count; 
  private Node dummy; 
  private AtomicReference<SmartPtr> head; 
  private AtomicReference<SmartPtr> tail; 
  
  // Semaphores to implement atomic CAS
  private Semaphore tailSema; 
  private Semaphore headSema; 
  private Semaphore curTailSema; 

  public LockFreeQueue() {
	// implement your constructor here
	this.count = new AtomicInteger(0); 
	this.dummy = new Node(-1); 
	this.head = new AtomicReference<>(); 
	this.tail = new AtomicReference<>(); 
	this.head.set(new SmartPtr(dummy));
	this.tail.set(new SmartPtr(dummy));
	
	this.tailSema = new Semaphore(1); 
	this.headSema = new Semaphore(1); 
	this.curTailSema = new Semaphore(1); 
  }

  public boolean enq(Integer value) {
	// implement your enq method here
	SmartPtr newNode = new SmartPtr(new Node(value)); 
	AtomicReference<SmartPtr> curTail = new AtomicReference<>(); 
	SmartPtr curNext; 
	
	while (true) {
		curTail.set(new SmartPtr(tail.get())); 
		curNext = new SmartPtr(curTail.get().ptr.next);
		if (curTail.get().EqualTo(tail.get())) {
			if (curNext.ptr == null) {
//				if (curTail.compareAndSet(
//					new SmartPtr(new Node(curTail.get().ptr.value, curNext.ptr), curNext.cnt++), 
//					new SmartPtr(new Node(curTail.get().ptr.value, newNode.ptr), newNode.cnt))) 
//					break; 
				try {
					curTailSema.acquire();
					if (curTail.get().ptr.next == curNext.ptr) {
						curTail.get().ptr.next = newNode.ptr; 
						curNext.cnt++; 
						break; 
					}
				} catch (InterruptedException e) {} finally {
					curTailSema.release(); 
				}
			}
			else {
//				tail.compareAndSet(curTail.get().GetAndIncCnt(), curNext); 
				try {
					tailSema.acquire();
					if (tail.get().EqualTo(curTail.get().GetAndIncCnt())) {
						tail.get().Set(curNext); 
					}
				} catch (InterruptedException e) {} finally {
					tailSema.release(); 
				} 
			}
		}
	}
	if (tail.compareAndSet(curTail.get().GetAndIncCnt(), newNode)) {
		count.getAndIncrement(); 
		return true; 
	}
	
    return false;
  }
  
  public Integer deq() {
	// implement your deq method here
	SmartPtr curHead; 
	SmartPtr curTail; 
	SmartPtr curNext;
	Integer res; 
	
	while (true) {
		curHead = new SmartPtr(head.get());
		curTail = new SmartPtr(tail.get());
		curNext = new SmartPtr(curHead.ptr.next);
		if (curHead.EqualTo(head.get())) {
			if (curHead.ptr == curTail.ptr) {
				if (curNext.ptr == null) 
					continue; 
				else {
//					tail.compareAndSet(curTail.GetAndIncCnt(), curNext); 
					try {
						tailSema.acquire();
						if (tail.get().EqualTo(curTail.GetAndIncCnt())) {
							tail.get().Set(curNext); 
						}
					} catch (InterruptedException e) {} finally {
						tailSema.release(); 
					} 
				}
			}
			else {
				res = curNext.ptr.value;
//				if (head.compareAndSet(curHead.GetAndIncCnt(), curNext)) 
//					break; 
				try {
					headSema.acquire();
					if (head.get().EqualTo(curHead.GetAndIncCnt())) {
						head.get().Set(curNext);
						break; 
					}
				} catch (InterruptedException e) {} finally {
					headSema.release(); 
				}
			}
		}
	}
    return res;
  }
  
  protected class SmartPtr {
	  public Node ptr; 
	  public Integer cnt; 
	  
	  public SmartPtr(Node p, Integer c) {
		  ptr = p; 
		  cnt = c; 
	  }
	  
	  public SmartPtr(SmartPtr that) {
		  ptr = that.ptr; 
		  cnt = that.cnt; 
	  }
	  
	  public SmartPtr(Node p) {
		  ptr = p; 
		  cnt = 0; 
	  }
	  
	  public SmartPtr GetAndIncCnt() {
		  SmartPtr res = new SmartPtr(this); 
		  cnt++; 
		  return res; 
	  }
	  
	  public void Set(SmartPtr that) {
		  ptr = that.ptr; 
		  cnt = that.cnt; 
	  }
	  
	  public boolean EqualTo(SmartPtr that) {
		  return ptr == that.ptr && cnt == that.cnt; 
	  }
  }
  
  protected class Node {
	  public Integer value;
	  public Node next;
		    
	  public Node(Integer x) {
		  value = x;
		  next = null;
	  }
	  
	  /* Added by Chunheng */ 
	  public Node (Integer x, Node y) {
		  value = x; 
		  next = y; 
	  }
  }
}
