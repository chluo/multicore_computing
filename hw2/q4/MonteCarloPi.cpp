#include <iostream> 
#include <cstdlib> 
#include <ctime> 
#include <omp.h>
#include <random> 

#define NUM_THREAD 8

/* This program requires c++11 */

using std::cout; 


class RandomPoint {
private: 
  double x; 
  double y; 

  double genRandom(void) {
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0); 
    return dist(gen); 
  }

public: 
  RandomPoint(void) {
    x = genRandom() - 0.5; 
    y = genRandom() - 0.5; 
  }

  bool isInCircle(void) {
    return (x* x + y* y < 0.5* 0.5); 
  }

  void printPoint(void) { 
    cout << '(' << x << ", " << y << ')' << '\n'; 
  }
}; 

double MonteCarloPi(int s) { 
  int c = 0;
  int i; 

  omp_set_num_threads(NUM_THREAD); 
  #pragma omp parallel for shared(c) private(i) 
  for (i = 0; i < s; i++) {
    // printf("Thread ID: %d\n", omp_get_thread_num());  
    if (RandomPoint().isInCircle()) 
      #pragma omp critical  
      c++; 
  }

  return 4* (double)c/ s; 
}

int main(void) {
  double start = omp_get_wtime(); 
  cout << MonteCarloPi(1000000) << '\n'; 
  double end   = omp_get_wtime(); 
  cout << "Execution Time: " << (end - start) << '\n'; 

  return 0; 
}
