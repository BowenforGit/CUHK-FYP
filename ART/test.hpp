#include <stdlib.h>    // malloc, free
#include <string.h>    // memset, memcpy
#include <stdint.h>    // integer types
#include <emmintrin.h> // x86 SSE intrinsics
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>  // gettime
#include <algorithm>   // std::random_shuffle
#include <vector>
#include <map>

struct Node;
struct leafNode;
void loadKey(uintptr_t tid,uint8_t key[]);
uint64_t getLeafKey(leafNode* leaf);
leafNode* lookup(Node* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength);
uint64_t getSmallestKey(Node* node);
uint64_t getLargestKey(Node* node);
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
  return ((double)now_tv.tv_sec) + ((double)now_tv.tv_usec)/1000000.0; // seconds
}

static inline uint64_t min(uint64_t a, uint64_t b) { // overload
   return (a<b)?a:b;
}

static inline uint64_t max(uint64_t a, uint64_t b) { // overload
   return (a>b)?a:b;
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
    // printf("keys[0]=%llu, keys[1]=%llu\n", keys[0], keys[1]);
  }
    
  if (mode==2) {
    // "pseudo-sparse" (the most-significant leaf bit gets lost)
    for (uint64_t i=0;i<n;i++) {
       keys[i]=(static_cast<uint64_t>(rand())<<32) | static_cast<uint64_t>(rand()); /* rand() return an int */
    }
  }

  // #ifdef DEBUG
  //   for(uint64_t i = 0; i < n; i++) {
  //     printf("keys[%llu]=%llu\n", i, keys[i]);
  //   }
  // #endif
}

void generate_values(uint64_t* values, uint64_t n) {
  for (uint64_t i=0;i<n;i++) {
      values[i]=i+1;
   }
  
  // std::random_shuffle(values,values+n);
}

void generate_without_set(uint64_t* keys, uint64_t n, uint64_t card, std::vector<uint64_t>& vec) {
  for(uint64_t i = 0; i < card; i++) {
    int idx = rand()%n;
    vec.push_back(keys[idx]);
  }
}

void erase_all(Node* tree, Node** nodeRef, uint64_t* keys, uint64_t n) {
  for (uint64_t i=0;i<n;i++) {
      uint8_t key[8];loadKey(keys[i],key);
      erase(tree,&tree,key,8,0,8);
  }
  // printf("Erase all elements\n");
  assert(tree == NULL);
}

Node* test_art_insertion(uint64_t* keys, uint64_t* values, uint64_t n, bool erase) {
  double start = gettime();
  Node* tree=NULL;
  for (uint64_t i=0;i<n;i++) {
    uint8_t key[8];loadKey(keys[i],key);
    insert(tree,&tree,key,0,values[i],8);
  }

  // #ifdef DEBUG
  //   for(uint64_t i = 0; i < n; i++) {
  //     uint8_t key[8]; loadKey(keys[i], key);
  //     printf("Look for key = %llu, 0x%llx\n", keys[i], keys[i]);
  //     leafNode* r = lookup(tree, key, 8, 0, 8);
  //     assert(r != NULL);
  //     uint64_t key_at_r = getLeafKey(r);
  //     printf("Find key = %llu; real key = %llu\n", key_at_r, keys[i]);
  //     assert(key_at_r == keys[i]);
  //   }
  // #endif

  if (erase) {
    // printf("insert,%lld,%f\n",n,(n/1000000.0)/(gettime()-start));
    printf("%f\n",(n/1000000.0)/(gettime()-start));
    erase_all(tree, &tree, keys, n);
  }
  return tree;
}

Node* test_art_bulk_loading(uint64_t* keys, uint64_t* values, uint64_t n, bool erase) {
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

  // #ifdef DEBUG_BULK
  //   for(uint64_t i = 0; i < n; i++) {
  //     uint8_t key[8]; loadKey(keys[i], key);
  //     printf("Look for key = %llu, 0x%llx\n", keys[i], keys[i]);
  //     leafNode* r = lookup(tree, key, 8, 0, 8);
  //     assert(r != NULL);
  //     uint64_t key_at_r = getLeafKey(r);
  //     printf("Find key = %llu; real key = %llu\n", key_at_r, keys[i]);
  //     assert(key_at_r == keys[i]);
  //   }
  // #endif

  if (erase) {
    // printf("bulk load,%lld,%f\n",n,(n/1000000.0)/(gettime()-start));
    printf("%f\n",(n/1000000.0)/(gettime()-start));
    erase_all(tree, &tree, keys, n);
  }

  return tree;
}

void test_art_lookup(uint64_t* keys, uint64_t* values, uint64_t n) { // WITHIN, EQ
  Node* tree = test_art_insertion(keys, values, n, false);
  
  double start = gettime();
  for(uint64_t i = 0; i < n; i++) {
    uint8_t key[8]; loadKey(keys[i], key);
    lookup(tree, key, 8, 0, 8);
  }
  // printf("art lookup,%lld,%f\n",n,(n/1000000.0)/(gettime()-start));
  printf("%f\n",(n/1000000.0)/(gettime()-start));


  erase_all(tree, &tree, keys, n);
}

void test_art_range_query(uint64_t* keys, uint64_t* values, uint64_t n, uint64_t width) { // LT, LTE, GT, GTE, BETWEEN, INSIDE, OUTSIDE, NEQ
  Node* tree = test_art_insertion(keys, values, n, false);

  std::vector<uint64_t> results;
  // randomly choose a range of length width
  int start_idx = rand()%n;
  uint64_t left = keys[start_idx];
  uint64_t right = left+width-1;
  
  double start = gettime();
  rangeQuery(tree, left, right, 8, results, 0, true, 0);

// #ifdef DEBUG
  printf("number of elements found = %lu\n", results.size());
  // for(auto item:results) {
  //   printf("%llu\n", item);
  // }
  // printf("range query,%lld,time=%f\n",n,gettime()-start);
// #endif
  
  printf("%f\n", gettime()-start);

  erase_all(tree, &tree, keys, n);
}

void test_art_WITHOUT(uint64_t* keys, uint64_t* values, uint64_t n, std::vector<uint64_t>& without) {
  Node* tree = test_art_insertion(keys, values, n, false);

  std::vector<uint64_t> results;
  
  double start = gettime();
  std::sort(without.begin(),without.end());
  uint64_t smallest = getSmallestKey(tree);
  // printf("smallest key = %llu\n", smallest);
  uint64_t largest = getLargestKey(tree);
  // printf("largest key = %llu\n", largest);
  rangeQuery(tree, smallest, max(without[0]-1,smallest), 8, results, 0, true, 0);
  // printf("range=[%llu,%llu], length=%llu, size = %lu\n", smallest, max(without[0]-1,smallest), (max(without[0]-1,smallest)-smallest+1),results.size());
  for (uint64_t i = 0; i < without.size()-1; i++) {
    rangeQuery(tree, without[i]+1, without[i+1]-1, 8, results, 0, true, 0);
    // printf("range=[%llu,%llu], length=%llu, size = %lu\n", without[i]+1, without[i+1]-1, (without[i+1]-1-without[i]),results.size());
  }
  rangeQuery(tree, min(without.back()+1, largest), largest, 8, results, 0, true, 0);
  // printf("range=[%llu,%llu], length=%llu, size = %lu\n", min(without.back()+1, largest), largest, (largest-min(without.back()+1, largest)+1),results.size());

  // printf("result set size = %lu\n", results.size());

  // printf("art without query,%lld,time=%f\n",n,gettime()-start);
  printf("%f\n", gettime()-start);

  erase_all(tree, &tree, keys, n);
}

void test_grasper_insertion(uint64_t* keys, uint64_t* values, uint64_t n) {
  double start = gettime();
  std::map<uint64_t, std::vector<uint64_t>> index_map;
  for (uint64_t i = 0; i < n; i++) {
    index_map[keys[i]].push_back(values[i]);
  }
  // printf("grasper insertion,%lld,%f\n",n,(n/1000000.0)/(gettime()-start));
  printf("%f\n",(n/1000000.0)/(gettime()-start));
}

void test_grasper_insertion(uint64_t* keys, uint64_t* values, uint64_t n, std::map<uint64_t, std::vector<uint64_t>>& index_map) {
  double start = gettime();
  for (uint64_t i = 0; i < n; i++) {
    index_map[keys[i]].push_back(values[i]);
  }
  // printf("grasper insertion,%lld,%f\n",n,(n/1000000.0)/(gettime()-start));
}

void test_grasper_lookup(uint64_t* keys, uint64_t* values, uint64_t n) {
  std::map<uint64_t, std::vector<uint64_t>> index_map;
  test_grasper_insertion(keys, values, n, index_map);
  std::vector<uint64_t> results;
  
  double start = gettime();
  for (uint64_t i = 0; i < n; i++) {
    auto itr = index_map.find(keys[i]);
    if (itr != index_map.end()) {
      results.insert(results.end(), itr->second.begin(), itr->second.end());
    }
  }
  // printf("grasper lookup,%lld,%f\n",n,(n/1000000.0)/(gettime()-start));
  printf("%f\n",(n/1000000.0)/(gettime()-start));
}

void test_grasper_range_query(uint64_t* keys, uint64_t* values, uint64_t n, uint64_t width) {
  std::map<uint64_t, std::vector<uint64_t>> index_map;
  test_grasper_insertion(keys, values, n, index_map);
  std::vector<uint64_t> results;

  int start_idx = rand()%n;
  uint64_t left = keys[start_idx];
  uint64_t right = left+width-1;

  double start = gettime();
  auto itr_low = index_map.lower_bound(left);
  auto itr_high = index_map.upper_bound(right);
  for (auto itr = itr_low; itr != itr_high; itr++) {
    results.insert(results.end(), itr->second.begin(), itr->second.end());
  }
  // printf("%lu elements found\n", results.size());
  // printf("grasper range query,%lld,time=%f\n",n,(gettime()-start));
  printf("%f\n", gettime()-start);
}

void test_grasper_WITHOUT(uint64_t* keys, uint64_t* values, uint64_t n, std::vector<uint64_t>& without) {
  std::map<uint64_t, std::vector<uint64_t>> index_map;
  test_grasper_insertion(keys, values, n, index_map);
  std::vector<uint64_t> results;

  double start = gettime();
  for (auto item : index_map) {
    auto itr = std::find(without.begin(), without.end(), item.first);
    if (itr == without.end()) {
      results.insert(results.end(), item.second.begin(), item.second.end());
    }
  }
  // printf("result set size = %lu\n", results.size());
  // printf("grasper without query,%lld,time=%f\n",n,(gettime()-start));
  printf("%f\n", gettime()-start);
}
