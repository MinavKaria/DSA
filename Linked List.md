# [21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)

## Approach 1: Using Vector

*Time Complexity:* $O(log n)$
*Space Complexity* : $O(2n+2m)$

```cpp
class Solution

{

ListNode* newNody(int key)

{

    ListNode* newNode = new ListNode;

    newNode->next=NULL;

    newNode->val=key;

    return newNode;

  

}

public:

    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2)

    {
        vector<int> arr;

        ListNode* temp1=list1;

        ListNode* temp2=list2;

  
        while(temp1!=NULL)
        {
            arr.push_back(temp1->val);

            temp1=temp1->next;
        }

         while(temp2!=NULL)
        {
            arr.push_back(temp2->val);
            temp2=temp2->next;
        }

        sort(arr.begin(),arr.end());

        ListNode* new_head=NULL;
        ListNode* new_tail=NULL;

  
        for(int i=0;i<arr.size();i++)
        {
             ListNode* temp=newNody(arr[i]);
            if(new_head==NULL)
            {
                new_head=temp;
                new_tail=temp;
            }

            else
            {
                new_tail->next=temp;
                new_tail=temp;
            }

        }
        return new_head;

    }

};
```


## Approach 2: Iterative , In Place


1. Point to the head to the first node of either` list1` or` list2 ` to `head` and make list1 iterate within that linked list only and make the list1 or list2 pointer to next where not checked and then head a new pointer `curr `on that head to traverse further.
2. After this if the `list1->val< list2 ->val` then 
	1.  curr->next=list1;
	2. list1=list1->next;
3. else
	1.  curr->next=list2;
	2. list2=list2->next;

4. Then `curr= curr ->next`
5. if list1 or list2 not complete then make that list next of `curr`

```cpp
class Solution {

public:

    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2)

    {

        // if list1 happen to be NULL
        // we will simply return list2.

        if(list1 == NULL)
            return list2;

        // if list2 happen to be NULL
        // we will simply return list1.
        
        if(list2 == NULL)
            return list1;

        ListNode * head = list1;
        if(list1 -> val >= list2 -> val)
        {
            head = list2;
            list2 = list2 -> next;
        }
        else
        {
            list1 = list1 -> next;
        }

        ListNode *curr = head;
        // till one of the list doesn't reaches NULL
        while(list1!=NULL &&  list2!=NULL)
        {
            if(list1 -> val < list2 -> val)
            {
                curr->next = list1;
                list1 = list1 -> next;
            }

            else
            {
                curr->next = list2;
                list2 = list2 -> next;
            }
            curr = curr -> next;
        }
        // adding remaining elements of bigger list.
        if(!list1)
            curr -> next = list2;
        else
            curr -> next = list1;

        return head;

    }

};
```



# [2807. Insert Greatest Common Divisors in Linked List](https://leetcode.com/problems/insert-greatest-common-divisors-in-linked-list/)

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution 
{
int gcd(int a,int b)
{
        if(b==0)
        {
            return a;
        }
        else
        {
            return gcd(b, a % b);
        }
}
ListNode* newNody(int key)
{
    ListNode *newNode=new ListNode;
    newNode->next=NULL;
    newNode->val=key;
    return newNode;
}
public:
    ListNode* insertGreatestCommonDivisors(ListNode* head) 
    {
        struct ListNode *ptr=NULL;
        struct ListNode *ptr1=NULL;
        ptr=head;
        ptr1=head->next;
        int gcda;
        while(ptr!=NULL || ptr1!=NULL)
        {
            if(ptr1!=NULL && ptr!=NULL)
            {
                gcda=gcd(ptr->val,ptr1->val);
            }
            else
            {
                break;
            }
            struct ListNode *newNode=newNody(gcda);
            newNode->next=ptr1;
            ptr->next=newNode;
            ptr=ptr->next->next;
            ptr1=ptr1->next;
        }
            
        return head;
    }
};
```