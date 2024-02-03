#include <stdint.h>

typedef unsigned long long int markPtr;
#define levelMax 32

// define skipNode class
class skipNode
{
public:
  int levelTop;
  markPtr key;
  markPtr nextSN[levelMax + 1];

  __device__ __host__ markPtr CreateMarkedReference(skipNode *rf, bool mk)
  {
    markPtr newMR = (markPtr)rf;
    newMR = newMR | mk;
    return newMR;
  }

// setter
  __device__ __host__ void SetNextNode(int i, skipNode *rf, bool mk)
  {
    nextSN[i] = CreateMarkedReference(rf, mk);
  }

// getter with rf
  __device__ skipNode *GetNextNode(int i)
  {
    markPtr rf = nextSN[i];
    return (skipNode *)((rf >> 1) << 1);
  }

// getter with rf and mark
  __device__ skipNode *GetMarkedNode(int i, bool *marked)
  {
    marked[0] = nextSN[i] % 2;
    return (skipNode *)((nextSN[i] >> 1) << 1);
  }

// atomic comp and swap next skipnode
  __device__ bool CompareAndSwapNextNode(int i, skipNode *oldrf, skipNode *newrf, bool oldmk, bool newmk)
  {
    markPtr oldMR = (markPtr)oldrf | oldmk;
    markPtr newMR = (markPtr)newrf | newmk;
    markPtr nextMR = atomicCAS(&(nextSN[i]), oldMR, newMR);
    if (nextMR == oldMR)
      return true;
    return false;
  }

  // construct boundaries head/tail skipnode
  skipNode(markPtr k)
  {
    key = k;
    levelTop = levelMax;
    for (int i = 0; i < levelMax + 1; i++)
    {
      nextSN[i] = CreateMarkedReference((skipNode *)NULL, false);
    }
  }
};