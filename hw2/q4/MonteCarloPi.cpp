#include <iostream> 
#include <cstdlib> 
#include <ctime> 

using std::cout; 

class RandomPoint {
private: 
  double x; 
  double y; 

  double genRandom(void) {
    return (double)rand() / (RAND_MAX + 1.0); 
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
  srand(time(0)); 
  int c = 0;
  int i; 

  #pragma omp parallel for shared(c) private(i)
  for (i = 0; i < s; i++) {
    if (RandomPoint().isInCircle()) 
      #pragma omp critical  
      c++; 
  }

  return 4* (double)c/ s; 
}

int main(void) {
  clock_t start, end; 
  start = clock(); 
  cout << MonteCarloPi(1000000) << '\n'; 
  end   = clock(); 
  cout << "Execution Time: " << (double)(end - start) << '\n'; 

  return 0; 
}
