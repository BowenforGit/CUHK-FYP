#include <stdlib.h>    // malloc, free
#include <string.h>    // memset, memcpy
#include <stdint.h>    // integer types
#include <emmintrin.h> // x86 SSE intrinsics
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>  // gettime
#include <algorithm>   // std::random_shuffle
#include <vector>

struct Node;
struct leafNode;
void loadKey(uintptr_t tid,uint8_t key[]);
leafNode* lookup(Node* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength);
void insert(Node* node,Node** nodeRef,uint8_t key[],unsigned depth,uintptr_t value,unsigned maxKeyLength);
void erase(Node* node,Node** nodeRef,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength);
Node* bulkLoad(uint8_t** keyMat, uint64_t* values, 
               const unsigned keyLen, const uint64_t start, 
               const uint64_t end, unsigned depth, unsigned prefixLen);

void rangeQuery(Node* node, 
               const uint64_t start, 
               const uint64_t end, 
               const unsigned keyLength,
               std::vector<uint64_t>& results, // vector of tid
               unsigned depth, 
               bool explore, 
               uint64_t matchedValue);

static double gettime(void) {
  struct timeval now_tv;
  gettimeofday (&now_tv,NULL);
  return ((double)now_tv.tv_sec) + ((double)now_tv.tv_usec)/1000000.0;
}

void generate_keys(uint64_t* keys, uint64_t n, int mode) {
  // Generate keys
   for (uint64_t i=0;i<n;i++){
      // dense, sorted
      keys[i]=i+1;
      // printf("key = %llu\n", keys[i]);  
   }
      
   if (mode==1) {
      // dense, random
      std::random_shuffle(keys,keys+n);
   }
      
   if (mode==2) {
      // "pseudo-sparse" (the most-significant leaf bit gets lost)
      for (uint64_t i=0;i<n;i++) {
         keys[i]=(static_cast<uint64_t>(rand())<<32) | static_cast<uint64_t>(rand()); /* rand() return an int */
      }
   }
}

void generate_values(uint64_t* values, uint64_t n) {
  for (uint64_t i=0;i<n;i++) {
      values[i]=i+1;
   }
  
  std::random_shuffle(values,values+n);
}

void erase_all(Node* tree, Node** nodeRef, uint64_t* keys, uint64_t n) {
  for (uint64_t i=0;i<n;i++) {
      uint8_t key[8];loadKey(keys[i],key);
      erase(tree,&tree,key,8,0,8);
  }
  printf("Erase all elements\n");
  assert(tree == NULL);
}

Node* test_insertion(uint64_t* keys, uint64_t* values, uint64_t n, bool erase) {
  double start = gettime();
  Node* tree=NULL;
  for (uint64_t i=0;i<n;i++) {
    uint8_t key[8];loadKey(keys[i],key);
    insert(tree,&tree,key,0,values[i],8);
  }
  printf("insert,%lld,%f\n",n,(n/1000000.0)/(gettime()-start));

  if (erase) {
    erase_all(tree, &tree, keys, n);
  }
  return tree;
}

Node* test_bulk_loading(uint64_t* keys, uint64_t* values, uint64_t n, bool erase) {
  uint8_t** keyMat = new uint8_t*[n];
  for(uint64_t i = 0; i < n; i++) {
    keyMat[i] = new uint8_t[8];
  }

  Node* tree = NULL;
  std::sort(keys, keys+n);
  double start = gettime();
  for (size_t i = 0; i < n; i++) {
    loadKey(keys[i], keyMat[i]); // store each key in a byte-wise manner
  }
  
  tree = bulkLoad(keyMat, values, 8, 0, n, 0, 0);
  printf("bulk load,%lld,%f\n",n,(n/1000000.0)/(gettime()-start));

  if (erase) {
    erase_all(tree, &tree, keys, n);
  }

  return tree;
}

void test_range_query(uint64_t* keys, uint64_t* values, uint64_t n, uint64_t left, uint64_t right) {
  Node* tree = test_insertion(keys, values, n, false);

  std::vector<uint64_t> results;
  
  double start = gettime();
  rangeQuery(tree, left, right, 8, results, 0, true, 0);
  printf("range query,%lld,time=%f\n",n,gettime()-start);

  erase_all(tree, &tree, keys, n);
}


