package queue;

import java.util.concurrent.atomic.*;

public class LockFreeQueue implements MyQueue {
	private AtomicInteger count; 
	private Node dummy; 
	private AtomicStampedReference<Node> head; 
	private AtomicStampedReference<Node> tail; 	
	public LockFreeQueue() {
		this.count = new AtomicInteger(0); 
		this.dummy = new Node(-1); 
		this.head = new AtomicStampedReference<>(dummy, 0); 
		this.tail = new AtomicStampedReference<>(dummy, 0); 
	}
	public boolean enq(Integer value) {
		AtomicStampedReference<Node> newNode = new AtomicStampedReference<>(new Node(value), 0); 
		AtomicStampedReference<Node> curTail = new AtomicStampedReference<>(null, 0); 
		AtomicStampedReference<Node> curNext = new AtomicStampedReference<>(null, 0); 
		while (true) {
			curTail.set(tail.getReference(), tail.getStamp());
			curNext.set(curTail.getReference().next.getReference(), curTail.getReference().next.getStamp());
			if (curTail.getReference() == tail.getReference() && curTail.getStamp() == tail.getStamp()) {
				if (curNext.getReference() == null) {
					if (curTail.getReference().next.compareAndSet(curNext.getReference(), newNode.getReference(), curNext.getStamp(), curNext.getStamp() + 1))
						break;
				}
				else 
					tail.compareAndSet(curTail.getReference(), curNext.getReference(), curTail.getStamp(), curTail.getStamp() + 1); 
			}
		}
		if (tail.compareAndSet(curTail.getReference(), newNode.getReference(), curTail.getStamp(), curTail.getStamp() + 1)) {
			count.getAndIncrement(); 
			return true; 
		}
		return false; 
	}	
	public Integer deq() {
		AtomicStampedReference<Node> curHead = new AtomicStampedReference<>(null, 0);
		AtomicStampedReference<Node> curTail = new AtomicStampedReference<>(null, 0);
		AtomicStampedReference<Node> curNext = new AtomicStampedReference<>(null, 0);
		int res; 
		while (true) {
			curHead.set(head.getReference(), head.getStamp());
			curTail.set(tail.getReference(), tail.getStamp());
			curNext.set(curHead.getReference().next.getReference(), curHead.getReference().next.getStamp());
			if (curHead.getReference() == head.getReference() && curHead.getStamp() == head.getStamp()) {
				if (curHead.getReference() == curTail.getReference()) {
					if (curNext.getReference() == null) 
						continue; 
					tail.compareAndSet(curTail.getReference(), curNext.getReference(), curTail.getStamp(), curTail.getStamp() + 1); 
				}
				else {
					res = curNext.getReference().value.intValue(); 
					if (head.compareAndSet(curHead.getReference(), curNext.getReference(), curHead.getStamp(), curHead.getStamp() + 1)) 
						break; 
				}
			}
		}
		return res; 
	}
	protected class Node {
		public Integer value;
		public AtomicStampedReference<Node> next;
		public Node(Integer x) {
			value = x;
			next = new AtomicStampedReference<>(null, 0);
		}
	}
/* 
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
	   
	  public Node (Integer x, Node y) {
		  value = x; 
		  next = y; 
	  }
  }
*/
}
