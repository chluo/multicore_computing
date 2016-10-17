package q3;

import q3.Garden.Benjamin;
import q3.Garden.Mary;
import q3.Garden.Newton;

/* 
 * Main class to test Garden
 * Created by Chunheng Luo
 */ 

public class Main {
	public static void main (String args[]) {
		Garden   garden = new Garden(5); 
		Newton   digger = new Newton(garden); 
		Benjamin seeder = new Benjamin(garden); 
		Mary     filler = new Mary(garden); 
		
		Thread   threadDigger = new Thread(digger); 
		Thread   threadSeeder = new Thread(seeder); 
		Thread   threadFiller = new Thread(filler); 
		
		threadDigger.start();
		threadSeeder.start();
		threadFiller.start(); 
	}
}