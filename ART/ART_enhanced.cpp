/*
  Adaptive Radix Tree
  Viktor Leis, 2012
  leis@in.tum.de
 */

#include <stdlib.h>    // malloc, free
#include <string.h>    // memset, memcpy
#include <stdint.h>    // integer types
#include <emmintrin.h> // x86 SSE intrinsics
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>  // gettime
#include <algorithm>   // std::random_shuffle
#include <vector>

// Constants for the node types
static const int8_t NodeType4=0;
static const int8_t NodeType16=1;
static const int8_t NodeType48=2;
static const int8_t NodeType256=3;

// The maximum prefix length for compressed paths stored in the
// header, if the path is longer it is loaded from the database on
// demand
static const unsigned maxPrefixLength=9;

// Shared header of all inner nodes
struct Node {
   // length of the compressed path (prefix)
   uint32_t prefixLength;
   // number of non-null children
   uint16_t count;
   // node type
   int8_t type;
   // compressed path (prefix)
   uint8_t prefix[maxPrefixLength];

   Node(int8_t type) : prefixLength(0),count(0),type(type) {}
};

// Node with up to 4 children
struct Node4 : Node {
   uint8_t key[4];
   Node* child[4];

   Node4() : Node(NodeType4) {
      memset(key,0,sizeof(key));
      memset(child,0,sizeof(child));
   }
};

// Node with up to 16 children
struct Node16 : Node {
   uint8_t key[16];
   Node* child[16];

   Node16() : Node(NodeType16) {
      memset(key,0,sizeof(key));
      memset(child,0,sizeof(child));
   }
};

static const uint8_t emptyMarker=48;

// Node with up to 48 children
struct Node48 : Node {
   uint8_t childIndex[256];
   Node* child[48];

   Node48() : Node(NodeType48) {
      memset(childIndex,emptyMarker,sizeof(childIndex));
      memset(child,0,sizeof(child));
   }
};

// Node with up to 256 children
struct Node256 : Node {
   Node* child[256];

   Node256() : Node(NodeType256) {
      memset(child,0,sizeof(child));
   }
};

struct leafNode {
   std::vector<uint64_t>* ptr; // a pointer to a vector of values
   unsigned key_len;
   uint32_t count;
   uint8_t* key;

   leafNode (std::vector<uint64_t>* v_ptr, unsigned key_len_, uint8_t* key_) :ptr(v_ptr), key_len(key_len_), count(0), key(NULL) {
      key = new uint8_t[key_len_];
      memcpy(key, key_, key_len_);
      // printf("new leaf node has key 0x%x%x%x%x%x%x%x%x\n", key[0],key[1],key[2],key[3],key[4],key[5],key[6],key[7]);
      // printf("Built a new leaf node. key = 0x%x%x%x%x%x%x%x%x.\n", key[0],key[1],key[2],key[3],key[4],key[5],key[6],key[7]);
   }

   void addValue (uint64_t tid) {
      ptr->push_back(tid);
      // printf("add value: %llu under key = 0x%x%x%x%x%x%x%x%x. leaf addr = %p\n", tid, key[0],key[1],key[2],key[3],key[4],key[5],key[6],key[7], this);
      count++;
   }

   unsigned checkKeyMatching(uint8_t* rhs, unsigned depth) {
      // printf("inside the matching function, key=0x%x%x%x%x%x%x%x%x\n", key[0],key[1],key[2],key[3],key[4],key[5],key[6],key[7]);
      unsigned pos = 0;
      while (depth + pos < key_len && key[depth+pos] == rhs[depth+pos]) {
         pos++;
      }
      return pos; // depth+pos is the position where mismatch happens; if no mismatch, depth+pos = key_len
   }

   ~leafNode() {
      delete ptr; 
      delete [] key;
   }
}; // a dedicated leaf node


// deprecated
inline uintptr_t getLeafValue(Node* node) {
   // The the value stored in the pseudo-leaf
   return reinterpret_cast<uintptr_t>(node)>>1;
}

inline Node* leaf2Child(leafNode* l) {
   uintptr_t p = reinterpret_cast<uintptr_t>(l);
   return reinterpret_cast<Node*>(p | 1);
}

inline leafNode* child2Leaf(Node* child) {
   uintptr_t p = reinterpret_cast<uintptr_t>(child);
   return reinterpret_cast<leafNode*>(p-1);
}


inline bool isLeaf(Node* node) {
   // Is the node a leaf?
   return reinterpret_cast<uintptr_t>(node)&1;
}

inline Node* makeLeaf(uint8_t key[], unsigned key_len, uintptr_t value) {
   // // Create a pseudo-leaf
   // return reinterpret_cast<Node*>((tid<<1)|1);

   // initialize a new leaf, return a pointer to the new leaf
   std::vector<uint64_t>* v_ptr = new std::vector<uint64_t>();
   leafNode* leaf = new leafNode(v_ptr, key_len, key);
   leaf->addValue(value);
   return leaf2Child(leaf);
}

inline Node* makeLeaf(uint8_t key[], unsigned key_len, uintptr_t value, uint64_t* keys) { // for dubugging
   // // Create a pseudo-leaf
   // return reinterpret_cast<Node*>((tid<<1)|1);

   // initialize a new leaf, return a pointer to the new leaf
   // printf("makeleaf 1: keys[0]=%llu\n", keys[0]);
   std::vector<uint64_t>* v_ptr = new std::vector<uint64_t>();
   // printf("makeleaf 2: keys[0]=%llu\n", keys[0]);
   leafNode* leaf = new leafNode(v_ptr, key_len, key);
   // printf("makeleaf 3: keys[0]=%llu\n", keys[0]);
   leaf->addValue(value);
   // printf("makeleaf 4: keys[0]=%llu\n", keys[0]);
   return leaf2Child(leaf);
}

uint8_t flipSign(uint8_t keyByte) {
   // Flip the sign bit, enables signed SSE comparison of unsigned values, used by Node16
   return keyByte^128;
}

void loadKey(uintptr_t tid,uint8_t key[]) {
   // Store the key of the tuple into the key vector
   // Implementation is database specific
   reinterpret_cast<uint64_t*>(key)[0]=__builtin_bswap64(tid); /* tid is little-endian, therefore we need to reverse the byte order */
}

// This address is used to communicate that search failed
Node* nullNode=NULL;

static inline unsigned ctz(uint16_t x) {
   // Count trailing zeros, only defined for x>0
#ifdef __GNUC__
   return __builtin_ctz(x);
#else
   // Adapted from Hacker's Delight
   unsigned n=1;
   if ((x&0xFF)==0) {n+=8; x=x>>8;}
   if ((x&0x0F)==0) {n+=4; x=x>>4;}
   if ((x&0x03)==0) {n+=2; x=x>>2;}
   return n-(x&1);
#endif
}

Node** findChild(Node* n,uint8_t keyByte) {
   // Find the next child for the keyByte
   switch (n->type) {
      case NodeType4: {
         Node4* node=static_cast<Node4*>(n);
         for (unsigned i=0;i<node->count;i++)
            if (node->key[i]==keyByte)
               return &node->child[i];
         return &nullNode;
      }
      case NodeType16: {
         Node16* node=static_cast<Node16*>(n);
         __m128i cmp=_mm_cmpeq_epi8(_mm_set1_epi8(flipSign(keyByte)),_mm_loadu_si128(reinterpret_cast<__m128i*>(node->key)));
         unsigned bitfield=_mm_movemask_epi8(cmp)&((1<<node->count)-1);
         if (bitfield)
            return &node->child[ctz(bitfield)]; else
            return &nullNode;
      }
      case NodeType48: {
         Node48* node=static_cast<Node48*>(n);
         if (node->childIndex[keyByte]!=emptyMarker)
            return &node->child[node->childIndex[keyByte]]; else
            return &nullNode;
      }
      case NodeType256: {
         Node256* node=static_cast<Node256*>(n);
         return &(node->child[keyByte]);
      }
   }
   throw; // Unreachable
}

leafNode* minimum(Node* node) {
   // Find the leaf with smallest key
   if (!node)
      return NULL;

   if (isLeaf(node))
      // return node;
      return child2Leaf(node);

   switch (node->type) {
      case NodeType4: {
         Node4* n=static_cast<Node4*>(node);
         return minimum(n->child[0]);
      }
      case NodeType16: {
         Node16* n=static_cast<Node16*>(node);
         return minimum(n->child[0]);
      }
      case NodeType48: {
         Node48* n=static_cast<Node48*>(node);
         unsigned pos=0;
         while (n->childIndex[pos]==emptyMarker)
            pos++;
         return minimum(n->child[n->childIndex[pos]]);
      }
      case NodeType256: {
         Node256* n=static_cast<Node256*>(node);
         unsigned pos=0;
         while (!n->child[pos])
            pos++;
         return minimum(n->child[pos]);
      }
   }
   throw; // Unreachable
}

leafNode* maximum(Node* node) {
   // Find the leaf with largest key
   if (!node)
      return NULL;

   if (isLeaf(node))
      // return node;
      return child2Leaf(node);

   switch (node->type) {
      case NodeType4: {
         Node4* n=static_cast<Node4*>(node);
         return maximum(n->child[n->count-1]);
      }
      case NodeType16: {
         Node16* n=static_cast<Node16*>(node);
         return maximum(n->child[n->count-1]);
      }
      case NodeType48: {
         Node48* n=static_cast<Node48*>(node);
         unsigned pos=255;
         while (n->childIndex[pos]==emptyMarker)
            pos--;
         return maximum(n->child[n->childIndex[pos]]);
      }
      case NodeType256: {
         Node256* n=static_cast<Node256*>(node);
         unsigned pos=255;
         while (!n->child[pos])
            pos--;
         return maximum(n->child[pos]);
      }
   }
   throw; // Unreachable
}

uint64_t getSmallestKey(Node* node) {
   leafNode* leaf = minimum(node);
   return __builtin_bswap64(*reinterpret_cast<uint64_t*>(leaf->key));
}

uint64_t getLargestKey(Node* node) {
   leafNode* leaf = maximum(node);
   return __builtin_bswap64(*reinterpret_cast<uint64_t*>(leaf->key));
}

// bool leafMatches(Node* leaf,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength) {
//    // Check if the key of the leaf is equal to the searched key
//    if (depth!=keyLength) {
//       uint8_t leafKey[maxKeyLength];
//       loadKey(getLeafValue(leaf),leafKey);
//       for (unsigned i=depth;i<keyLength;i++)
//          if (leafKey[i]!=key[i])
//             return false;
//    }
//    return true;
// }

bool leafMatches(leafNode* leaf, uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength) {
   // Check if the key of the leaf is equal to the searched key
   if (depth != keyLength) {
      // uint8_t leafKey[maxKeyLength];
      // loadKey(getLeafValue(leaf),leafKey);
      for (unsigned i=depth;i<keyLength;i++)
         if (leaf->key[i]!=key[i])
            return false;
   }
   return true;
}

unsigned prefixMismatch(Node* node,uint8_t key[],unsigned depth,unsigned maxKeyLength) {
   // Compare the key with the prefix of the node, return the number matching bytes
   unsigned pos;
   if (node->prefixLength>maxPrefixLength) { // optimistic PC
      for (pos=0;pos<maxPrefixLength;pos++)
         if (key[depth+pos]!=node->prefix[pos])
            return pos;
      // uint8_t minKey[maxKeyLength];
      // loadKey(getLeafValue(minimum(node)),minKey);
      leafNode* minLeaf = minimum(node);
      uint8_t* minKey = minLeaf->key;
      for (;pos<node->prefixLength;pos++)
         if (key[depth+pos]!=minKey[depth+pos])
            return pos;
   } else { // pessimistic PC
      for (pos=0;pos<node->prefixLength;pos++)
         if (key[depth+pos]!=node->prefix[pos])
            return pos;
   }
   return pos;
}

leafNode* lookup(Node* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength) {
   // Find the node with a matching key, optimistic version

   bool skippedPrefix=false; // Did we optimistically skip some prefix without checking it?

   while (node!=NULL) {
      // printf("Input address: %p\n", node);
      if (isLeaf(node)) {
         leafNode* leaf = child2Leaf(node);
         if (!skippedPrefix&&depth==keyLength) // No check required
            return leaf;

         if (depth!=keyLength) {
            // Check leaf
            uint8_t* leafKey = leaf->key;
            // loadKey(getLeafValue(node),leafKey);
            for (unsigned i=(skippedPrefix?0:depth);i<keyLength;i++)
               if (leafKey[i]!=key[i]) {
                  return NULL;
               }
         }
         return leaf;
      }

      // printf("it is not a leaf node, node type = %d, prefix length = %d\n", node->type, node->prefixLength);

      if (node->prefixLength) {
         // printf("check prefix lenghth here\n");
         // printf("0x%x%x%x%x%x%x%x%x\n", key[0],key[1],key[2],key[3],key[4],key[5],key[6],key[7]);
         if (node->prefixLength<maxPrefixLength) {
            // printf("key length = %u, prefix length = %u, depth = %u\n", keyLength, node->prefixLength, depth);
            for (unsigned pos=0;pos<node->prefixLength;pos++) {
               if (key[depth+pos]!=node->prefix[pos]){
                  return NULL;
               }     
            }
         } else {
            skippedPrefix=true;
         }
         
         depth+=node->prefixLength;
      }
      node=*findChild(node,key[depth]);
      depth++;
   }
   // printf("look up return here!\n");

   return NULL;
}


// Forward references
void insertNode4(Node4* node,Node** nodeRef,uint8_t keyByte,Node* child);
void insertNode16(Node16* node,Node** nodeRef,uint8_t keyByte,Node* child);
void insertNode48(Node48* node,Node** nodeRef,uint8_t keyByte,Node* child);
void insertNode256(Node256* node,Node** nodeRef,uint8_t keyByte,Node* child);

unsigned min(unsigned a,unsigned b) {
   // Helper function
   return (a<b)?a:b;
}

void copyPrefix(Node* src,Node* dst) {
   // Helper function that copies the prefix from the source to the destination node
   dst->prefixLength=src->prefixLength;
   memcpy(dst->prefix,src->prefix,min(src->prefixLength,maxPrefixLength));
}

void insert(Node* node,Node** nodeRef,uint8_t key[],unsigned depth,uintptr_t value,unsigned maxKeyLength) {
   // Insert the leaf value into the tree

   if (node==NULL) { // no previous child here
      *nodeRef=makeLeaf(key, maxKeyLength, value);
      // printf("1: keys[0]=%llu\n", keys[0]);
      return;
   }

   if (isLeaf(node)) { /* a leaf is a node pointer that stores a value */

      /* New implementation starts */
      leafNode* leaf = child2Leaf(node);
      // check if the prefix matches
      unsigned newPrefixLength = leaf->checkKeyMatching(key, depth);
         // printf("here\n");
      if (depth+newPrefixLength == maxKeyLength) { // duplicate keys
         leaf->addValue(value); // append the new value to the leaf
         // printf("2: keys[0]=%llu\n", keys[0]);
         return;

      } else { // not duplicate keys
         Node4* newNode=new Node4();
         newNode->prefixLength=newPrefixLength;
         memcpy(newNode->prefix,key+depth,min(newPrefixLength,maxPrefixLength)); /* Update the prefix (could be longer) */
         *nodeRef=newNode;
         insertNode4(newNode,nodeRef,leaf->key[depth+newPrefixLength],node /* use pointer to store value */);
         // printf("3.0: keys[0]=%llu\n", keys[0]); // ok
         Node* newLeaf = makeLeaf(key, maxKeyLength, value);
         // printf("3.1: keys[0]=%llu\n", keys[0]); 
         insertNode4(newNode,nodeRef,key[depth+newPrefixLength], newLeaf /* use pointer to store value */);
         // printf("3.2: keys[0]=%llu\n", keys[0]);

         return;
      }
      /* New implementation ends*/

      // // Replace leaf with Node4 and store both leaves in it
      // uint8_t existingKey[maxKeyLength];
      // loadKey(getLeafValue(node),existingKey); /* The key that has already been in the leaf */
      // unsigned newPrefixLength=0;
      // while (existingKey[depth+newPrefixLength]==key[depth+newPrefixLength]) /* at least the first "depth" byte must be the same */
      //    newPrefixLength++;

      // Node4* newNode=new Node4();
      // newNode->prefixLength=newPrefixLength;
      // memcpy(newNode->prefix,key+depth,min(newPrefixLength,maxPrefixLength)); /* Update the prefix (could be longer) */
      // *nodeRef=newNode;

      // insertNode4(newNode,nodeRef,existingKey[depth+newPrefixLength],node /* use pointer to store value */);
      // insertNode4(newNode,nodeRef,key[depth+newPrefixLength],makeLeaf(value) /* use pointer to store value */);
      // return;
   }

   // Handle prefix of inner node
   if (node->prefixLength) {
      unsigned mismatchPos=prefixMismatch(node,key,depth,maxKeyLength);
      if (mismatchPos!=node->prefixLength) {
         // Prefix differs, create new node
         Node4* newNode=new Node4();
         *nodeRef=newNode;
         newNode->prefixLength=mismatchPos;
         memcpy(newNode->prefix,node->prefix,min(mismatchPos,maxPrefixLength));
         // printf("4: keys[0]=%llu\n", keys[0]);

         // Break up prefix
         if (node->prefixLength<maxPrefixLength) {
            insertNode4(newNode,nodeRef,node->prefix[mismatchPos],node);
            node->prefixLength-=(mismatchPos+1);
            memmove(node->prefix,node->prefix+mismatchPos+1,min(node->prefixLength,maxPrefixLength));
         } else {
            node->prefixLength-=(mismatchPos+1);
            // uint8_t minKey[maxKeyLength];
            // loadKey(getLeafValue(minimum(node)),minKey);

            leafNode* minLeaf = minimum(node);
            uint8_t* minKey = minLeaf->key;
            insertNode4(newNode,nodeRef,minKey[depth+mismatchPos],node);
            memmove(node->prefix,minKey+depth+mismatchPos+1,min(node->prefixLength,maxPrefixLength));
         }
         insertNode4(newNode,nodeRef,key[depth+mismatchPos],makeLeaf(key, maxKeyLength, value));
         return;
      }
      depth+=node->prefixLength;
      // printf("5: keys[0]=%llu\n", keys[0]);

   }

   // Recurse
   Node** child=findChild(node,key[depth]);
   if (*child) {
      insert(*child,child,key,depth+1,value,maxKeyLength);
      return;
   }

   // Insert leaf into inner node
   Node* newNode=makeLeaf(key, maxKeyLength, value);
   switch (node->type) {
      case NodeType4: insertNode4(static_cast<Node4*>(node),nodeRef,key[depth],newNode); break;
      case NodeType16: insertNode16(static_cast<Node16*>(node),nodeRef,key[depth],newNode); break;
      case NodeType48: insertNode48(static_cast<Node48*>(node),nodeRef,key[depth],newNode); break;
      case NodeType256: insertNode256(static_cast<Node256*>(node),nodeRef,key[depth],newNode); break;
   }
}

void insertNode4(Node4* node,Node** nodeRef,uint8_t keyByte,Node* child) {
   // Insert leaf into inner node
   if (node->count<4) {
      // Insert element
      unsigned pos;
      for (pos=0;(pos<node->count)&&(node->key[pos]<keyByte);pos++); /* Ensure the order */
      memmove(node->key+pos+1,node->key+pos,node->count-pos); /* Move the key array */
      memmove(node->child+pos+1,node->child+pos,(node->count-pos)*sizeof(uintptr_t)); /* Move the pointer array */
      node->key[pos]=keyByte;
      node->child[pos]=child;
      node->count++;
   } else {
      // Grow to Node16
      Node16* newNode=new Node16();
      *nodeRef=newNode;
      newNode->count=4;
      copyPrefix(node,newNode);
      for (unsigned i=0;i<4;i++)
         newNode->key[i]=flipSign(node->key[i]);
      memcpy(newNode->child,node->child,node->count*sizeof(uintptr_t));
      delete node;
      return insertNode16(newNode,nodeRef,keyByte,child);
   }
}

void insertNode16(Node16* node,Node** nodeRef,uint8_t keyByte,Node* child) {
   // Insert leaf into inner node
   if (node->count<16) {
      // Insert element
      uint8_t keyByteFlipped=flipSign(keyByte);
      __m128i cmp=_mm_cmplt_epi8(_mm_set1_epi8(keyByteFlipped),_mm_loadu_si128(reinterpret_cast<__m128i*>(node->key)));
      uint16_t bitfield=_mm_movemask_epi8(cmp)&(0xFFFF>>(16-node->count));
      unsigned pos=bitfield?ctz(bitfield):node->count;
      memmove(node->key+pos+1,node->key+pos,node->count-pos);
      memmove(node->child+pos+1,node->child+pos,(node->count-pos)*sizeof(uintptr_t));
      node->key[pos]=keyByteFlipped;
      node->child[pos]=child;
      node->count++;
   } else {
      // Grow to Node48
      Node48* newNode=new Node48();
      *nodeRef=newNode;
      memcpy(newNode->child,node->child,node->count*sizeof(uintptr_t));
      for (unsigned i=0;i<node->count;i++)
         newNode->childIndex[flipSign(node->key[i])]=i;
      copyPrefix(node,newNode);
      newNode->count=node->count;
      delete node;
      return insertNode48(newNode,nodeRef,keyByte,child);
   }
}

void insertNode48(Node48* node,Node** nodeRef,uint8_t keyByte,Node* child) {
   // Insert leaf into inner node
   if (node->count<48) {
      // Insert element
      unsigned pos=node->count;
      if (node->child[pos])
         for (pos=0;node->child[pos]!=NULL;pos++);
      node->child[pos]=child;
      node->childIndex[keyByte]=pos;
      node->count++;
   } else {
      // Grow to Node256
      Node256* newNode=new Node256();
      for (unsigned i=0;i<256;i++)
         if (node->childIndex[i]!=48)
            newNode->child[i]=node->child[node->childIndex[i]];
      newNode->count=node->count;
      copyPrefix(node,newNode);
      *nodeRef=newNode;
      delete node;
      return insertNode256(newNode,nodeRef,keyByte,child);
   }
}

void insertNode256(Node256* node,Node** nodeRef,uint8_t keyByte,Node* child) {
   // Insert leaf into inner node
   node->count++;
   node->child[keyByte]=child;
}

// Forward references
void eraseNode4(Node4* node,Node** nodeRef,Node** leafPlace);
void eraseNode16(Node16* node,Node** nodeRef,Node** leafPlace);
void eraseNode48(Node48* node,Node** nodeRef,uint8_t keyByte);
void eraseNode256(Node256* node,Node** nodeRef,uint8_t keyByte);

void erase(Node* node,Node** nodeRef,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength) {
   // Delete a leaf from a tree
   // printf("start erasing\n");
   if (!node) // null node
      return;

   if (isLeaf(node)) {
      // Make sure we have the right leaf
      leafNode* leaf = child2Leaf(node);
      if (leafMatches(leaf,key,keyLength,depth,maxKeyLength)) {
         *nodeRef=NULL;
         delete leaf;
      }
      return;
   }

   // Handle prefix
   if (node->prefixLength) {
      if (prefixMismatch(node,key,depth,maxKeyLength)!=node->prefixLength)
         return;
      depth+=node->prefixLength;
   }

   Node** child=findChild(node,key[depth]);
   if (isLeaf(*child)&&leafMatches(child2Leaf(*child),key,keyLength,depth,maxKeyLength)) {
      // Leaf found, delete it in inner node
      switch (node->type) {
         case NodeType4: eraseNode4(static_cast<Node4*>(node),nodeRef,child); break;
         case NodeType16: eraseNode16(static_cast<Node16*>(node),nodeRef,child); break;
         case NodeType48: eraseNode48(static_cast<Node48*>(node),nodeRef,key[depth]); break;
         case NodeType256: eraseNode256(static_cast<Node256*>(node),nodeRef,key[depth]); break;
      }
   } else {
      //Recurse
      erase(*child,child,key,keyLength,depth+1,maxKeyLength);
   }
}

void eraseNode4(Node4* node,Node** nodeRef,Node** leafPlace) {
   // Delete leaf from inner node
   // printf("Erase Node4\n");
   unsigned pos=leafPlace-node->child;
   memmove(node->key+pos,node->key+pos+1,node->count-pos-1);
   memmove(node->child+pos,node->child+pos+1,(node->count-pos-1)*sizeof(uintptr_t));
   node->count--;

   if (node->count==1) {
      // Get rid of one-way node
      Node* child=node->child[0];
      if (!isLeaf(child)) {
         // Concantenate prefixes
         unsigned l1=node->prefixLength;
         if (l1<maxPrefixLength) {
            node->prefix[l1]=node->key[0];
            l1++;
         }
         if (l1<maxPrefixLength) {
            unsigned l2=min(child->prefixLength,maxPrefixLength-l1);
            memcpy(node->prefix+l1,child->prefix,l2);
            l1+=l2;
         }
         // Store concantenated prefix
         memcpy(child->prefix,node->prefix,min(l1,maxPrefixLength));
         child->prefixLength+=node->prefixLength+1;
      }
      *nodeRef=child;
      delete node;
   }
}

void eraseNode16(Node16* node,Node** nodeRef,Node** leafPlace) {
   // Delete leaf from inner node
   // printf("Erase Node16\n");
   unsigned pos=leafPlace-node->child;
   memmove(node->key+pos,node->key+pos+1,node->count-pos-1);
   memmove(node->child+pos,node->child+pos+1,(node->count-pos-1)*sizeof(uintptr_t));
   node->count--;

   if (node->count==3) {
      // Shrink to Node4
      Node4* newNode=new Node4();
      newNode->count=node->count;
      copyPrefix(node,newNode);
      for (unsigned i=0;i<4;i++)
         newNode->key[i]=flipSign(node->key[i]);
      memcpy(newNode->child,node->child,sizeof(uintptr_t)*4);
      *nodeRef=newNode;
      delete node;
   }
}

void eraseNode48(Node48* node,Node** nodeRef,uint8_t keyByte) {
   // Delete leaf from inner node
   // printf("Erase Node48\n");
   node->child[node->childIndex[keyByte]]=NULL;
   node->childIndex[keyByte]=emptyMarker;
   node->count--;

   if (node->count==12) {
      // Shrink to Node16
      Node16 *newNode=new Node16();
      *nodeRef=newNode;
      copyPrefix(node,newNode);
      for (unsigned b=0;b<256;b++) {
         if (node->childIndex[b]!=emptyMarker) {
            newNode->key[newNode->count]=flipSign(b);
            newNode->child[newNode->count]=node->child[node->childIndex[b]];
            newNode->count++;
         }
      }
      delete node;
   }
}

void eraseNode256(Node256* node,Node** nodeRef,uint8_t keyByte) {
   // Delete leaf from inner node
   // printf("Erase Node256\n");
   node->child[keyByte]=NULL;
   node->count--;

   if (node->count==37) {
      // Shrink to Node48
      Node48 *newNode=new Node48();
      *nodeRef=newNode;
      copyPrefix(node,newNode);
      for (unsigned b=0;b<256;b++) {
         if (node->child[b]) {
            newNode->childIndex[b]=newNode->count;
            newNode->child[newNode->count]=node->child[b];
            newNode->count++;
         }
      }
      delete node;
   }
}


inline uint64_t min(uint64_t a, uint64_t b) { // overload
   return (a<b)?a:b;
}

// keyMat[i][depth] is the byte concerned
Node* bulkLoad(uint8_t** keyMat, uint64_t* values, 
               const unsigned keyLen, const uint64_t start, 
               const uint64_t end, unsigned depth, unsigned prefixLen) {
   // return condition
   // static int cnt = 0;
   // printf("here: %d\n", ++cnt);
   if (depth == keyLen) {
      // return NULL;
      leafNode* leaf = child2Leaf(makeLeaf(keyMat[start], keyLen, values[start]));
      for (uint64_t i = start+1; i < end; i++) {
         leaf->addValue(values[i]);
      }
      return leaf2Child(leaf);
   }
   
   uint64_t localKeyCount = end - start;

   if (localKeyCount == 1) { // lazy expansion
      return makeLeaf(keyMat[start], keyLen, values[start]);
   }

   uint8_t distinctKeys[min(localKeyCount, 256)]; // distinct keys at this depth
   uint64_t distinctKeyCount = 0; // number of distinct keys
   uint64_t offset[min(localKeyCount, 256)+1]; // the start position of each distinc keys
   for (int i = start; i < end; i++) {
      if (i == start) {
         distinctKeys[0] = keyMat[i][depth];
         distinctKeyCount++;
         offset[0] = 0;
      } else if (distinctKeys[distinctKeyCount-1] != keyMat[i][depth]) {
         distinctKeys[distinctKeyCount] = keyMat[i][depth];
         offset[distinctKeyCount] = i-start;
         distinctKeyCount++;
      }
   }
   offset[distinctKeyCount] = end - start;
   assert(distinctKeyCount >= 1 && distinctKeyCount <= 256 && distinctKeyCount <= localKeyCount);

   if (distinctKeyCount == 1) { // localKeyCount > 1
      if (depth == keyLen-1) { // last level
         leafNode* leaf = child2Leaf(makeLeaf(keyMat[start], keyLen, values[start]));
         for (uint64_t i = start+1; i < end; i++) {
            leaf->addValue(values[i]);
         }
         return leaf2Child(leaf);
      } else { // path compression
         return bulkLoad(keyMat, values, 
                        keyLen, start, 
                        end, depth+1, prefixLen+1);
      }
   }

   // printf("Hello, start = %llu, end = %llu, #distinct keys = %llu, depth = %u\n", start, end, distinctKeyCount, depth);
   // build a local node
   if (distinctKeyCount <= 4) {
      Node4* newNode = new Node4();
      // check prefix
      if (prefixLen > 0) {
         if (prefixLen > maxPrefixLength) {
             newNode->prefixLength = prefixLen; // skip prefix, turn to optimistic
         } else {
            assert(depth - prefixLen < keyLen);
            memcpy(newNode->prefix, &(keyMat[start][depth-prefixLen]), prefixLen);
            newNode->prefixLength = prefixLen;
         }
      }
      for (int i = 0; i < distinctKeyCount; i++) {
         newNode->key[i] = distinctKeys[i];
         newNode->child[i] = bulkLoad(keyMat, values, 
                                    keyLen, start+offset[i], 
                                    start+offset[i+1], depth+1, 0); // prefixLen is 0 again
      }
      newNode->count = distinctKeyCount;
      return newNode;
   } 
   else if (distinctKeyCount > 4 && distinctKeyCount <= 16) {
      Node16* newNode = new Node16();
      // check prefix
      if (prefixLen > 0) {
         if (prefixLen > maxPrefixLength) {
             newNode->prefixLength = prefixLen;
         } else {
            assert(depth - prefixLen < keyLen);
            memcpy(newNode->prefix, &(keyMat[start][depth-prefixLen]), prefixLen);
            newNode->prefixLength = prefixLen;
         }
      }
      for (int i = 0; i < distinctKeyCount; i++) {
         newNode->key[i] = distinctKeys[i];
         newNode->child[i] = bulkLoad(keyMat, values, 
                                    keyLen, start+offset[i], 
                                    start+offset[i+1], depth+1, 0); // prefixLen is 0 again
      }
      newNode->count = distinctKeyCount;
      return newNode;
   } 
   else if (distinctKeyCount > 16 && distinctKeyCount <= 48) {
      Node48* newNode = new Node48();
      // check prefix
      if (prefixLen > 0) {
         if (prefixLen > maxPrefixLength) {
             newNode->prefixLength = prefixLen;
         } else {
            assert(depth - prefixLen < keyLen);
            memcpy(newNode->prefix, &(keyMat[start][depth-prefixLen]), prefixLen);
            newNode->prefixLength = prefixLen;
         }
      }
      for (int i = 0; i < distinctKeyCount; i++) {
         uint8_t keyByte = keyMat[start+offset[i]][depth];
         newNode->childIndex[keyByte] = i;
         newNode->child[i] = bulkLoad(keyMat, values, 
                                    keyLen, start+offset[i], 
                                    start+offset[i+1], depth+1, 0); // prefixLen is 0 again
      }
      newNode->count = distinctKeyCount;
      return newNode;
   } 
   else { // (48,256] 
      // printf("Node 256!\n");
      Node256* newNode = new Node256();
      // check prefix
      if (prefixLen > 0) {
         if (prefixLen > maxPrefixLength) {
             newNode->prefixLength = prefixLen;
         } else {
            // printf("Copying prefix\n");
            assert(depth - prefixLen < keyLen);
            memcpy(newNode->prefix, &(keyMat[start][depth-prefixLen]), prefixLen);
            newNode->prefixLength = prefixLen;

            // for (unsigned i = 0; i < prefixLen; i++) {
            //    printf("%u\n", newNode->prefix[i]);
            // }
         }
      }
      for (int i = 0; i < distinctKeyCount; i++) {
         uint8_t keyByte = keyMat[start+offset[i]][depth];
         // printf("keyByte = %u\n", keyByte);
         newNode->child[keyByte] = bulkLoad(keyMat, values, 
                                    keyLen, start+offset[i], 
                                    start+offset[i+1], depth+1, 0); // prefixLen is 0 again
      }
      newNode->count = distinctKeyCount;
      return newNode;
   }
}


// get all data that fall into [start, end]
void rangeQuery(Node* node, 
               const uint64_t start, 
               const uint64_t end, 
               const unsigned keyLength,
               std::vector<uint64_t>& results, // vector of tid
               unsigned depth, 
               bool explore, 
               uint64_t matchedValue) {

   if (node == NULL) return;
   if (start > end) // invalid input
      return;

   // assert(node->type);
   // printf("Call! explore=%u, address=%p, depth = %u\n", explore, node, depth);

   if (!explore && !isLeaf(node)) { // directly go to the next
      if (node->type == NodeType4) {
         // printf("Unexplored Node4 encountered, depth=%d, addr=%p\n", depth, node);
         for (unsigned i = 0; i < 4; i++) {
            rangeQuery(static_cast<Node4*>(node)->child[i], start, end, keyLength, results, depth+1, false, matchedValue);
         }
      }
      else if (node->type == NodeType16) {
         // printf("Unexplored Node16 encountered, depth=%d, addr=%p\n", depth,node);
         for (unsigned i = 0; i < node->count; i++) {
            rangeQuery(static_cast<Node16*>(node)->child[i], start, end, keyLength, results, depth+1, false, matchedValue);
         }
      }
      else if (node->type == NodeType48) {
         // printf("Unexplored Node48 encountered, depth=%d, addr=%p\n", depth, node);
         unsigned cnt = 0;
         Node48* tmp = static_cast<Node48*>(node);
         for (unsigned i = 0; i<48; i++) {
            if (tmp->childIndex[i] != emptyMarker) {
               // cnt++;
               rangeQuery(tmp->child[tmp->childIndex[i]], start, end, keyLength, results, depth+1, false, matchedValue);
            }
         }
      }
      else if (node->type == NodeType256) {
         // printf("Unexplored Node256 encountered, depth=%d\n", depth);
         Node256* tmp = static_cast<Node256*>(node);
         for (unsigned i = 0; i < 256; i++) {
            rangeQuery(tmp->child[i], start, end, keyLength, results, depth+1, false, matchedValue);
         }
         // printf("Unexplored Node256 encountered, depth=%d\n", depth);
      }
      return;
   }

   if (isLeaf(node)) {
      // uint64_t leafValue = getLeafValue(node);
      leafNode* leaf = child2Leaf(node);
      uint64_t leafKey =  __builtin_bswap64(*reinterpret_cast<uint64_t*>(leaf->key));
      if (depth == keyLength) { // no compression on the path
         // results.push_back(leafValue);
         for (uint32_t i = 0; i < leaf->count; i++) {
            results.push_back((*(leaf->ptr))[i]);
         }
      }
      else if (leafKey >= start && leafKey <= end) { // check
         // results.push_back(leafKey);
         // printf("%lu\n", results.size());
         for (uint32_t i = 0; i < leaf->count; i++) {
            // printf("Cond 2: Leaf node with key %llu encountered!\n", leafKey);
            results.push_back((*(leaf->ptr))[i]);
         }
      }
      return;
   }
   else {
      if (node->prefixLength > maxPrefixLength) { // a compressed node
         // printf("Inner node->compressed node\n");
         rangeQuery(node, start, end, keyLength, results, depth, false, matchedValue); // re-search this node and set the explore flag to false
         return;
      }
      else {
         // because it is still exploring, the value for the previous depths must match either start or end
         // printf("Inner node->uncompressed node\n");

         // update depth
         depth += node->prefixLength; // the index of the byte to be visited
         
         // compute prefix value
         uint64_t prefixValue = 0;
         for (unsigned i = 0; i < node->prefixLength; i++) {
            prefixValue  = (prefixValue << 8) + node->prefix[i];
         }

         // compute new matched value
         matchedValue = (matchedValue << node->prefixLength) + prefixValue;

         uint64_t left = start >> (8*(7-depth));
         uint64_t right = end >> (8*(7-depth));
         switch (node->type) {
            case(NodeType4): {
               Node4* tmp = static_cast<Node4*>(node);
               for (unsigned i = 0; i < tmp->count; i++) {
                  uint64_t value = (matchedValue << 8) + tmp->key[i];
                  if (value > left && value < right) {
                     // traverse all subtree
                     rangeQuery(tmp->child[i], start, end, keyLength, results, depth+1, false, value);
                  }
                  else if (value == left || value == right) {
                     rangeQuery(tmp->child[i], start, end, keyLength, results, depth+1, true, value);
                  }
                  else if (value > right) {
                     break; // for loop
                  }
               }
               break; // for case
            }

            case(NodeType16): {
               Node16* tmp = static_cast<Node16*>(node);
               for (unsigned i = 0; i < tmp->count; i++) {
                  uint64_t value = (matchedValue << 8) + tmp->key[i];
                  if (value > left && value < right) {
                     // traverse all subtree
                     rangeQuery(tmp->child[i], start, end, keyLength, results, depth+1, false, value);
                  }
                  else if (value == left || value == right) {
                     rangeQuery(tmp->child[i], start, end, keyLength, results, depth+1, true, value);
                  }
                  else if (value > right){
                     break; // for loop
                  }
               }
               break; // for case
            }

            case(NodeType48): {
               Node48* tmp = static_cast<Node48*>(node);
               unsigned cnt = 0;
               for (uint8_t i = 0; cnt < tmp->count; i++) {
                  if (tmp->childIndex[i] == emptyMarker) continue;
                  cnt++;
                  uint64_t value = (matchedValue << 8) + i;
                  if (value > left && value < right) {
                     // traverse all subtree
                     rangeQuery(tmp->child[tmp->childIndex[i]], start, end, keyLength, results, depth+1, false, value);
                  }
                  else if (value == left || value == right) {
                     rangeQuery(tmp->child[tmp->childIndex[i]], start, end, keyLength, results, depth+1, true, value);
                  }
                  else if (value > right) {
                     break; // for loop
                  }
               }
               break; // for case
            }

            case(NodeType256): {
               Node256* tmp = static_cast<Node256*>(node);
               for (unsigned i = 0; i < tmp->count; i++) {
                  uint64_t value = (matchedValue << 8) + i;
                  if (value > left && value < right) {
                     // traverse all subtree
                     rangeQuery(tmp->child[i], start, end, keyLength, results, depth+1, false, value);
                  }
                  else if (value == left || value == right) {
                     rangeQuery(tmp->child[i], start, end, keyLength, results, depth+1, true, value);
                  }
                  else if (value > right) {
                     break; // for loop
                  }
               }
               break; // for case
            }
         } // switch
      } // else
   } // else
}


static double gettime(void) {
  struct timeval now_tv;
  gettimeofday (&now_tv,NULL);
  return ((double)now_tv.tv_sec) + ((double)now_tv.tv_usec)/1000000.0;
}

// int main(int argc,char** argv) {
//    if (argc!=3) {
//       printf("usage: %s n 0|1|2\nn: number of keys\n0: sorted keys\n1: dense keys\n2: sparse keys\n", argv[0]);
//       return 1;
//    }

//    uint64_t n=atoi(argv[1]);
//    uint64_t* keys=new uint64_t[n];

//    // Generate keys
//    for (uint64_t i=0;i<n;i++){
//       // dense, sorted
//       keys[i]=i+1;
//       // printf("key = %llu\n", keys[i]);  
//    }
      
//    if (atoi(argv[2])==1) {
//       // dense, random
//       std::random_shuffle(keys,keys+n);
//    }
      
//    if (atoi(argv[2])==2) {
//       // "pseudo-sparse" (the most-significant leaf bit gets lost)
//       for (uint64_t i=0;i<n;i++) {
//          keys[i]=(static_cast<uint64_t>(rand())<<32) | static_cast<uint64_t>(rand()); /* rand() return an int */
//       }
//    }
   
//    // define value
//    uint64_t* values=new uint64_t[n];
//    memcpy(values, keys, 8*n);

//    // associate value with keys

//    /*=========================== Bulk load starts ===========================*/
   
//    uint8_t** keyMat = new uint8_t*[n];
//    for(uint64_t i = 0; i < n; i++) 
//       keyMat[i] = new uint8_t[8];

//    // size_t elemSize = sizeof(*keys);
//    double start = gettime();
//    std::sort(keys, keys+n);
//    for (size_t i = 0; i < n; i++) {
//       loadKey(keys[i], keyMat[i]); // store each key in a byte-wise manner
//    }
   
//    Node* tree = NULL;
//    tree = bulkLoad(keyMat, values, 8, 0, n, 0, 0);
//    // bulkLoad(uint8_t **keyMat, const uintptr_t *values, const unsigned int keyLen, const uint64_t start, const uint64_t end, unsigned int depth, unsigned int prefixLen)
//    printf("bulk load,%lld,%f\n",n,(n/1000000.0)/(gettime()-start)); // throughput

//    /*=========================== Bulk load ends ===========================*/



//    // Build tree
//    // keys[1] = 101;
//    // keys[2] = 101;
//    // keys[3] = 101;
//    // keys[4] = 101;
//    // keys[5] = 101;
//    // keys[6] = 101;
//    // keys[7] = 101;
//    // keys[8] = 101;
//    // keys[49] = 101;
//    // keys[53] = 101;
//    // keys[56] = 101;
//    // keys[67] = 101;
//    // keys[71] = 101;
//    // keys[89] = 101;
//    // double start = gettime();
//    // Node* tree=NULL;
//    // for (uint64_t i=0;i<n;i++) {
//    //    uint8_t key[8];loadKey(keys[i],key);
//    //    // printf("before inserting key %llu and value %llu, keys[0]=%llu\n", keys[i], values[i], keys[0]);
//    //    insert(tree,&tree,key,0,values[i],8);
//    //    // printf("after insertion, keys[0]=%llu\n", keys[0]);
//    // }
//    // printf("insert,%lld,%f\n",n,(n/1000000.0)/(gettime()-start));
   
//    delete[] values;

//    /*=========================== Lookup starts ===========================*/
//    // Repeat lookup for small trees to get reproducable results
//    // uint64_t repeat=10000000/n;
//    // if (repeat<1)
//    //    repeat=1;
//    uint64_t repeat = 1;
//    start = gettime();
//    for (uint64_t r=0;r<repeat;r++) {
//       for (uint64_t i=0;i<n;i++) {
//          uint8_t key[8];loadKey(keys[i],key);
//          // printf("look up key = %llu\n", keys[i]);
//          leafNode* leaf=lookup(tree,key,8,0,8);
//          printf("Find leaf address for key %llu: %p\n", keys[i], leaf);
//          // for (unsigned i = 0; i < leaf->count; i++) {
//          //    printf("Find %llu ", (*(leaf->ptr))[i]);
//          // }
//          // printf("return, leaf address = %p\n", leaf);
//          // printf("\n");
//          // assert(isLeaf(leaf) && getLeafValue(leaf)==keys[i]);
//          // assert((*(leaf->ptr))[0] == values[i]);
//       }
//    }
//    printf("lookup,%lld,%f\n",n,(n*repeat/1000000.0)/(gettime()-start));
//    /*=========================== Lookup ends ===========================*/


   
//    /*=========================== range query starts ===========================*/
//    // std::vector<uint64_t> results;
//    // rangeQuery(tree, 1, 50, 8, results, 0, true, 0);
//    // printf("Finish range query, data found = %lu\n", results.size());
//    // int cnt = 0;
//    // for(int i = 0; i < results.size(); i++) {
//    //    if (results[i] >= 1 && results[i] <=50) {
//    //       cnt++;
//    //    }
//    // }
//    // printf("Real count = %d\n", cnt);
//    /*=========================== range query ends ===========================*/

//    /*=========================== erase starts ===========================*/
//    start = gettime();
//    for (uint64_t i=0;i<n;i++) {
//       uint8_t key[8];loadKey(keys[i],key);
//       erase(tree,&tree,key,8,0,8);
//    }
//    printf("erase,%lld,%f\n",n,(n/1000000.0)/(gettime()-start));
//    assert(tree==NULL);
//    /*=========================== erase ends ===========================*/

//    return 0;
// }
