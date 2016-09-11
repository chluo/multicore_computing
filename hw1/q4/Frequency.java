import java.util.ArrayList;
import java.util.concurrent.*;
import java.util.Arrays;
/**
 * Created by wenwen on 9/6/16.
 */
public class Frequency implements Callable<Integer>{
    int x;
    int[] items;
    public Frequency(int x, int[] items){
        this.x = x;
        this.items = items;
    }
    public Integer call(){
        try{
            int frequency = 0;
            for(int item:items){
                if(item == x){
                    frequency++;
                }
            }
            return frequency;
        }catch (Exception e){
            System.err.println (e);
            return 1;
        }
    }
    
    public static int prarellelFreq(int x, int[] items, int numThreads){
        ExecutorService threadPool = Executors.newFixedThreadPool(numThreads);
        try{
            int frequency = 0;
            ArrayList<int[]> listOfItems = new ArrayList<>();
            ArrayList<Future<Integer>> results = new ArrayList<>();
            int len = items.length/numThreads;
            int startIndex = 0;
            int endIndex = len;
            while(startIndex != items.length) {
                int[] subArray = Arrays.copyOfRange(items, startIndex, endIndex);
                startIndex = endIndex;
                endIndex = endIndex + len;
                if(endIndex > items.length){
                    endIndex = items.length;
                }
                listOfItems.add(subArray);
            }

            for(int[] listOfItem : listOfItems){
                Future<Integer> t = threadPool.submit(new Frequency(x,listOfItem));
                results.add(t);
            }
            for(Future<Integer> result:results){
                frequency = frequency + result.get();
            }
            
            threadPool.shutdown();
            return frequency;
        }catch (Exception e){
            System.err.println(e);
            return 0;
        }
    }
}
