#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "SkipNode.cuh"

#define numOpsThr 64
#define threadsPerBlock 512

int ttlOps;
int key;

class skipNode;

// define skiplist class
class skiplistCUDA
{
public:
  skipNode *head;
  skipNode *tail;
  skiplistCUDA()
  {
    skipNode *hd = new skipNode(0);
    skipNode *tl = new skipNode((markPtr)0xffffffff);
    cudaMalloc((void **)&head, sizeof(skipNode));
    cudaMalloc((void **)&tail, sizeof(skipNode));
    for (int i = 0; i < hd->levelTop + 1; i++)
    {
      hd->SetNextNode(i, tail, false);
    }
    cudaMemcpy(head, hd, sizeof(skipNode), cudaMemcpyHostToDevice);

    cudaMemcpy(tail, tl, sizeof(skipNode), cudaMemcpyHostToDevice);
  }
  __device__ bool findHelper(markPtr, skipNode **, skipNode **); 
  __device__ bool INSERT(markPtr, int ttlops);
  __device__ bool DELETE(markPtr);
  __device__ bool CONTAIN(markPtr);
};

__device__ skipNode **skipNodes;
__device__ unsigned int pointerIndex = 0;
__device__ markPtr *randoms;

// create new skipnode in skiplist
__device__ skipNode *GetNewSN(markPtr key, int ttlops)
{
  markPtr ind = atomicInc(&pointerIndex, ttlops);
  skipNode *n = skipNodes[ind];
  n->key = key;
  n->levelTop = randoms[ind];
  for (int i = 0; i < n->levelTop + 1; i++)
  {
    n->SetNextNode(i, NULL, false);
  }
  return n;
}

// generate level for new skipnode
markPtr GenRandLevel()
{
  markPtr l = 1;
  double prob = 0.5;
  while (((rand() / (double)(RAND_MAX)) < prob) && (l < levelMax))
    l++;
  return l;
}

__device__ skiplistCUDA *List; 

__global__ void init(skiplistCUDA *ls, skipNode **sn, markPtr *r)
{
  randoms = r;
  skipNodes = sn;
  List = ls;
}

// helper function like contain() 
// traversing skiplist from head SN to bottom level
// snips out marked SN along the way
__device__ bool
skiplistCUDA::findHelper(markPtr searchKey, skipNode **preds, skipNode **succs)
{
  int levelBottom = 0;
  bool marked[] = {false};
  bool snip;
  skipNode *pred = NULL;
  skipNode *curr = NULL;
  skipNode *succ = NULL;
  bool isOperationComplete;
  while (true)
  {
    isOperationComplete = false;
    pred = head;
    for (int level = levelMax; level >= levelBottom; level--)
    {
      curr = pred->GetNextNode(level);
      while (true)
      {
        succ = curr->GetMarkedNode(level, marked);
        while (marked[0])
        {
          snip = pred->CompareAndSwapNextNode(level, curr, succ, false, false);
          isOperationComplete = true;
          if (!snip)
            break;
          curr = pred->GetNextNode(level);
          succ = curr->GetMarkedNode(level, marked);
          isOperationComplete = false;
        }
        if (isOperationComplete && !snip)
          break;
        if (curr->key < searchKey)
        {
          pred = curr;
          curr = succ;
        }
        else
        {
          break;
        }
      }
      if (isOperationComplete && !snip)
        break;
      preds[level] = pred;
      succs[level] = curr;
    }
    if (isOperationComplete && !snip)
      continue;
    return ((curr->key == searchKey));
  }
}

// traverse the skiplist like findHelper()
// Like findHelper(), contain() ignores keys of marked SN. Unlike findHelper(),
// it does not try to remove marked SN and jumps over them
__device__ bool
skiplistCUDA::CONTAIN(markPtr key)
{
  int levelBottom = 0;
  bool marked = false;
  skipNode *pred = head;
  skipNode *curr = NULL;
  skipNode *succ = NULL;
  for (int level = levelMax; level >= levelBottom; level--)
  {
    curr = pred->GetNextNode(level);
    while (true)
    {
      succ = curr->GetMarkedNode(level, &marked);
      while (marked)
      {
        curr = curr->GetNextNode(level);
        succ = curr->GetMarkedNode(level, &marked);
      }
      if (curr->key < key)
      {
        pred = curr;
        curr = succ;
      }
      else
      {
        break;
      }
    }
  }
  return (curr->key == key);
}

// uses findHelper() to check if an unmarked SN with key = deleteKey is in the bottom-level skiplist
// if yes then logically removes the deleteKey from the abstract set
// and prepares it for physical removal
__device__ bool
skiplistCUDA::DELETE(markPtr deleteKey)
{
  int levelBottom = 0;
  skipNode *preds[levelMax + 1];
  skipNode *succs[levelMax + 1];
  skipNode *succ;
  bool marked[] = {false};
  while (true)
  {
    bool isFound = findHelper(deleteKey, preds, succs);
    if (!isFound)
    {
      return false;
    }
    else
    {
      skipNode *nodeToDelete = succs[levelBottom];
      for (int level = nodeToDelete->levelTop; level >= levelBottom + 1; --level)
      {
        succ = nodeToDelete->GetMarkedNode(level, marked);
        while (!marked[0])
        {
          nodeToDelete->CompareAndSwapNextNode(level, succ, succ, false, true);
          succ = nodeToDelete->GetMarkedNode(level, marked);
        }
      }
      succ = nodeToDelete->GetMarkedNode(levelBottom, marked);
      while (true)
      {
        bool didMark = nodeToDelete->CompareAndSwapNextNode(levelBottom, succ, succ, false, true);
        succ = succs[levelBottom]->GetMarkedNode(levelBottom, marked);
        if (didMark)
        {
          findHelper(deleteKey, preds, succs);
          return true;
        }
        else if (marked[0])
        {
          return false;
        }
      }
    }
  }
}

// uses findHelper() to check if a node with key = insertKey is already in the skiplist
// if no add new SN with key = insertKey
__device__ bool
skiplistCUDA::INSERT(markPtr insertKey, int totalLevels)
{
  skipNode *newskipNode = GetNewSN(insertKey, totalLevels);
  int levelTop = newskipNode->levelTop;
  int levelBottom = 0;
  skipNode *preds[levelMax + 1];
  skipNode *succs[levelMax + 1];
  while (true)
  {
    bool isFound = findHelper(insertKey, preds, succs);
    if (isFound)
    {
      return false;
    }
    else
    {
      for (int level = levelBottom; level <= levelTop; ++level)
      {
        skipNode *succ = succs[level];
        newskipNode->SetNextNode(level, succ, false);
      }
      skipNode *pred = preds[levelBottom];
      skipNode *succ = succs[levelBottom];
      bool isSwapped;

      isSwapped = pred->CompareAndSwapNextNode(levelBottom, succ, newskipNode, false, false);
      if (!isSwapped)
      {
        continue;
      }
      for (int level = levelBottom + 1; level <= levelTop; ++level)
      {
        while (true)
        {
          pred = preds[level];
          succ = succs[level];
          if (pred->CompareAndSwapNextNode(level, succ, newskipNode, false, false))
          {
            break;
          }
          findHelper(insertKey, preds, succs);
        }
      }
      return true;
    }
  }
}

#define insert 0
#define delete 1
#define contain 2

// kernel to execute operation set of insert/delete/contain
__global__ void skiplistkernel(markPtr *keysVals, markPtr *opsList, markPtr *res, int ttlops, int opsPerTh)
{

  int threadId;
  for (int i = 0; i < opsPerTh; i++)
  {
    threadId = i * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= ttlops)
      return;

    markPtr item = keysVals[threadId];
    if (opsList[threadId] == insert)
    {
      res[threadId] = List->INSERT(item, ttlops);
    }
    if (opsList[threadId] == delete)
    {
      res[threadId] = List->DELETE(item);
    }
    if (opsList[threadId] == contain)
    {
      res[threadId] = List->CONTAIN(item);
    }
  }
}

// main
int main(int argc, char **argv)
{
 
  int Inserts = atoi(argv[1]);
  int deletes = atoi(argv[2]);
  int ttlOps = atoi(argv[3]);
  int key = atoi(argv[4]);

  markPtr *opsList = (markPtr *)malloc(sizeof(markPtr) * ttlOps);
  markPtr *levels = (markPtr *)malloc(sizeof(markPtr) * ttlOps);
  markPtr *keysVals = (markPtr *)malloc(sizeof(markPtr) * ttlOps);
  markPtr *res = (markPtr *)malloc(sizeof(markPtr) * ttlOps);

  srand(0);
  for (int i = 0; i < ttlOps; i++)
  {
    keysVals[i] = 20 + rand() % key;
  }

  srand(0);
  for (int i = 0; i < ttlOps; i++)
  {
    levels[i] = GenRandLevel() - 1;
  }

  int i;
  for (i = 0; i < (ttlOps * Inserts) / 100; i++)
  {
    opsList[i] = insert;
  }
  for (; i < (ttlOps * (Inserts + deletes)) / 100; i++)
  {
    opsList[i] = delete;
  }
  for (; i < ttlOps; i++)
  {
    opsList[i] = contain;
  }

  Inserts = (ttlOps * Inserts) / 100;

  markPtr *keysValsD;
  markPtr *opsListD;
  markPtr *resD;
  markPtr *levelsD;

  cudaMalloc((void **)&resD, sizeof(markPtr) * ttlOps);
  cudaMalloc((void **)&keysValsD, sizeof(markPtr) * ttlOps);
  cudaMalloc((void **)&opsListD, sizeof(markPtr) * ttlOps);
  cudaMalloc((void **)&levelsD, sizeof(markPtr) * ttlOps);
  cudaMemcpy(levelsD, levels, sizeof(markPtr) * ttlOps, cudaMemcpyHostToDevice);
  cudaMemcpy(keysValsD, keysVals, sizeof(markPtr) * ttlOps, cudaMemcpyHostToDevice);
  cudaMemcpy(opsListD, opsList, sizeof(markPtr) * ttlOps, cudaMemcpyHostToDevice);

  skipNode **pointers = (skipNode **)malloc(sizeof(markPtr) * Inserts);
  skipNode **Cpointers;

  for (int i = 0; i < Inserts; i++)
  {
    cudaMalloc((void **)&pointers[i], sizeof(skipNode));
  }
  cudaMalloc((void **)&Cpointers, sizeof(skipNode *) * Inserts);
  cudaMemcpy(Cpointers, pointers, sizeof(skipNode *) * Inserts, cudaMemcpyHostToDevice);

  skiplistCUDA *Clist;
  skiplistCUDA *list = new skiplistCUDA();
  cudaMalloc((void **)&Clist, sizeof(skiplistCUDA));
  cudaMemcpy(Clist, list, sizeof(skiplistCUDA), cudaMemcpyHostToDevice);

  int blocks = (ttlOps % (threadsPerBlock * numOpsThr) == 0) ? ttlOps / (threadsPerBlock * numOpsThr) : (ttlOps / (threadsPerBlock * numOpsThr)) + 1;

  init<<<1, 32>>>(Clist, Cpointers, levelsD);
  cudaDeviceSynchronize();

  cudaEvent_t startT, stopT;
  cudaEventCreate(&startT);
  cudaEventCreate(&stopT);
  cudaEventRecord(startT, 0);

  skiplistkernel<<<blocks, threadsPerBlock>>>(keysValsD, opsListD, resD, ttlOps, numOpsThr);

  cudaDeviceSynchronize();
  cudaEventRecord(stopT, 0);
  cudaEventSynchronize(stopT);
  float time;
  cudaEventElapsedTime(&time, startT, stopT);

  printf("%lf\n", time);

  cudaMemcpy(res, resD, sizeof(markPtr) * ttlOps, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  return 0;
}
