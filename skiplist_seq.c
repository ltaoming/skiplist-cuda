#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "skiplist_seq.h"

/* max level in the skip list is 8 */
#define MAX_LEVEL 8

/* genenrate random level for a node */
int genRandLevel(void)
{
    int level = 1;
    while ((rand() & 1) == 1 && level < MAX_LEVEL)
    {
        level++;
    }
    return level;
}

/* initialize skip-list */
void initSkipList(SkipList *sl)
{
    Node *head = (Node *)malloc(sizeof(Node));
    head->value = INT_MIN;
    head->level = MAX_LEVEL;
    head->next = (Node **)malloc(sizeof(Node *) * MAX_LEVEL);

    for (int i = 0; i < MAX_LEVEL; i++)
    {
        head->next[i] = NULL;
    }
    sl->head = head;
}

/* insert a value into the skip list */
void insert(SkipList *sl, int value)
{
    Node *destNode = sl->head;
    Node *newNode = (Node *)malloc(sizeof(Node));
    int level = genRandLevel();
    newNode->level = level;
    newNode->next = (Node **)malloc(sizeof(Node *) * MAX_LEVEL);
    newNode->value = value;

    // find the position of the newNode at each level
    for (int i = sl->head->level - 1; i >= 0; i--)
    {
        while (destNode->next[i] != NULL && destNode->next[i]->value < value)
        {
            destNode = destNode->next[i];
        }
        if (i < level)
        {
            newNode->next[i] = destNode->next[i];
            destNode->next[i] = newNode;
        }
    }
}

// 1 for contains, 0 for not contains
int contains(SkipList *sl, int value)
{
    Node *node = sl->head;
    Node *next = NULL;
    for (int i = sl->head->level - 1; i >= 0; i--)
    {
        while ((next = node->next[i]) && (next != NULL) && next->value < value)
        {
            node = next;
        }
        if (next && next->value == value)
            return 1;
    }
    return 0;
}

void delete(SkipList *sl, int value)
{
    Node *update[MAX_LEVEL];
    Node *node = sl->head;
    for (int i = sl->head->level - 1; i >= 0; i--)
    {
        while (node->next[i] != NULL && node->next[i]->value < value)
        {
            node = node->next[i];
        }
        update[i] = node;
    }
    node = node->next[0];
    if (node == NULL || node->value != value)
    {
        return;
    }
    for (int i = 0; i < node->level; i++)
    {
        if (update[i]->next[i] != node)
        {
            break;
        }
        update[i]->next[i] = node->next[i];
    }
    free(node->next);
    free(node);
}

void printSkipList(SkipList *sl)
{
    printf("SkipList:\n");
    for (int i = sl->head->level - 1; i >= 0; i--)
    {
        Node *node = sl->head->next[i];
        printf("Level %d: ", i);
        while (node != NULL)
        {
            printf("%d ", node->value);
            node = node->next[i];
        }
        printf("\n");
    }
}
