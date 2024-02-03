#include <limits.h>

/* 
node structure in skip-list:
value: Stores the value
next: Array of pointers. Each pointer points to the next node at each level in skip list.
For example, next[0] points to the next node in skip-list at level 0 (level 0 is the bottom level)
*/
typedef struct Node {
    int value;
    int level;
    struct Node **next;
} Node;

/* skip-list */
typedef struct SkipList {
    Node *head;
} SkipList;

/* Generic struct (Maybe implement this later) */
/* typedef struct Node {
    void *value;
    struct Node **next;
} Node;

typedef struct SkipList {
    int level;
    Node *header;
    int (*cmp)(void *, void *);
    size_t value_size;
} SkipList; */



/* Generate random level for a node */
int genRandLevel(void);

/* initialize skip list */
void initSkipList(SkipList *sl);

/* insert a value into the skip list */
void insert(SkipList *sl, int value);

/* check if a value is in the skip list */
int contains(SkipList *sl, int value);

/* delete a value from the skip list */
void delete(SkipList *sl, int value);

/* print the skip list */
void printSkipList(SkipList *sl);
