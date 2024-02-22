
## I would recommend you to  and Clone the repo and Download Obsidian and open this on it, for better experience





# DSA 

## Upper Bound  and Lower Bound 

```cpp

// lower_bound and upper_bound in vector

#include <algorithm> // for lower_bound, upper_bound and sort
#include <iostream>
#include <vector> // for vector

using namespace std;

int main()
{
	int gfg[] = { 5, 5, 5, 6, 6, 6, 7, 7 };

	vector<int> v(gfg, gfg + 8); // 5 5 5 6 6 6 7 7

	vector<int>::iterator lower, upper;
	lower = lower_bound(v.begin(), v.end(), 6);
	upper = upper_bound(v.begin(), v.end(), 6);

	cout << "lower_bound for 6 at index "
		<< (lower - v.begin()) << '\n';
	cout << "upper_bound for 6 at index "
		<< (upper - v.begin()) << '\n';

	return 0;
}

//Output:
lower_bound for 6 at index 3
upper_bound for 6 at index 6
```

```cpp
#include <bits/stdc++.h>
using namespace std;

int main()
{
	int n;
	cin >> n;
	int arr[n];

	for (int x = 0; x < n; x++) {
		cin >> arr[x];
	}
	auto itr = upper_bound(arr, arr + n, 6);
	cout << itr << endl; // returns the address position
						// which has value greater than 6
	cout << *itr << endl; // returns the no. stored at the
						// itr memory address
	auto it = upper_bound(arr, arr + n, 6)
			- arr; // returns the index position of itr
					// element in array
	cout << it << endl;
	auto itr2 = upper_bound(arr, arr + n,
							3); // gives the element which
								// has value greater than 3
	cout << *itr2;
	return 0;
}
```


```cpp
// C++ program to implement iterative Binary Search
#include <bits/stdc++.h>
using namespace std;

// An iterative binary search function.
int binarySearch(int arr[], int l, int r, int x)
{
	while (l <= r) {
		int m = l + (r - l) / 2;

		// Check if x is present at mid
		if (arr[m] == x)
			return m;

		// If x greater, ignore left half
		if (arr[m] < x)
			l = m + 1;

		// If x is smaller, ignore right half
		else
			r = m - 1;
	}

	// If we reach here, then element was not present
	return -1;
}

// Driver code
int main(void)
{
	int arr[] = { 2, 3, 4, 10, 40 };
	int x = 10;
	int n = sizeof(arr) / sizeof(arr[0]);
	int result = binarySearch(arr, 0, n - 1, x);
	(result == -1)
		? cout << "Element is not present in array"
		: cout << "Element is present at index " << result;
	return 0;
}

```


![[Pasted image 20231218140935.png]]
# [94. Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)

Preorder and Postorder are the same but only difference is
Preorder-> Root Left Right
Postorder-> Left Right Root
Inorder-> Left Root Right

```cpp
class Solution 
{
vector<int> inorder;
public:
    vector<int> inorderTraversal(TreeNode* root) 
    {
        if(root!=NULL)
        {
            inorderTraversal(root->left);
            inorder.push_back(root->val);
            inorderTraversal(root->right);
        }

        return inorder;
    }
};
```

# [872. Leaf-Similar Trees](https://leetcode.com/problems/leaf-similar-trees/)

```cpp
class Solution {
public:
    bool leafSimilar(TreeNode* root1, TreeNode* root2) {
        ios_base::sync_with_stdio(0),cin.tie(0),cout.tie(0);
        vector<int> a;
        leafArray(root1,a);
        vector<int> b;
        leafArray(root2,b);
        return a==b;
    }
    vector<int> leafArray(TreeNode* root,vector<int>& leafs){
        
        if(root!=NULL)
        {
            if(root->left==NULL && root->right==NULL)
            {
                leafs.push_back(root->val);
            }
            leafArray(root->left,leafs);
            leafArray(root->right,leafs);
        }
        return leafs;
    }
};
```

# [2385. Amount of Time for Binary Tree to Be Infected](https://leetcode.com/problems/amount-of-time-for-binary-tree-to-be-infected/)

# Introduction

Dynamic programming is a method for solving complex problems by breaking them down into simpler overlapping subproblems and solving each subproblem only once, storing the solutions to subproblems in a table to avoid redundant computations. The approach is particularly useful for optimization problems, where the goal is to find the best solution among a set of feasible solutions.

Key characteristics of dynamic programming include:

1. **Optimal Substructure:** An optimal solution to the problem can be constructed from optimal solutions of its subproblems.

2. **Overlapping Subproblems:** The problem can be broken down into smaller, overlapping subproblems, and the solutions to these subproblems are reused rather than recomputed.

Dynamic programming can be categorized into two main types:

### 1. **Top-down (Memoization) Dynamic Programming:**

In this approach, the problem is solved in a recursive manner, and the solutions to subproblems are stored in a data structure (often a table or a dictionary) to avoid redundant calculations. This method is also known as memoization.

**Example: Fibonacci Sequence**

The Fibonacci sequence can be computed using dynamic programming to avoid redundant calculations. The recursive approach would involve a lot of repeated calculations, but by storing the results of subproblems, you can significantly improve efficiency.

```python
def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]

# Example usage:
result = fibonacci(5)
print(result)  # Output: 5
```

### 2. **Bottom-up (Tabulation) Dynamic Programming:**

In this approach, the problem is solved in a systematic, iterative manner, starting from the simplest subproblems and building up to the original problem. The solutions are stored in a table.

**Example: Longest Common Subsequence**

Given two sequences, find the length of the longest subsequence present in both of them. This is a classic dynamic programming problem.

```python
def longest_common_subsequence(X, Y):
    m = len(X)
    n = len(Y)

    # Create a table to store results of subproblems
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Build the table in a bottom-up manner
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]

# Example usage:
X = "ABCBDAB"
Y = "BDCAB"
result = longest_common_subsequence(X, Y)
print(result)  # Output: 4
```

In this example, the table `dp` is filled in a bottom-up manner, and the result is found in the bottom-right corner of the table.

Dynamic programming is a powerful technique that can be applied to a wide range of problems, from simple to complex, and it often leads to more efficient algorithms compared to naive approaches.
# [935. Knight Dialer](https://leetcode.com/problems/knight-dialer/) 

```cpp
public:
    vector<vector<int>> memo;
    int n;
    int MOD = 1e9 + 7;


    vector<vector<int>> jumps = 
    {
        {4, 6},
        {6, 8},
        {7, 9},
        {4, 8},
        {3, 9, 0},
        {},
        {1, 7, 0},
        {2, 6},
        {1, 3},
        {2, 4}
    };
    
    int dp(int remain, int square) 
    {
        if (remain == 0) 
        {
            return 1;
        }
        
        if (memo[remain][square] != 0) 
        {
            return memo[remain][square];
        }
        
        int ans = 0;
        for (int nextSquare : jumps[square]) 
        {
            ans = (ans + dp(remain - 1, nextSquare)) % MOD;
        }
        
        memo[remain][square] = ans;
        return ans;
    }
    
    int knightDialer(int n) 
    {
        this->n = n;
        memo = vector(n+1, vector(10, 0));
        int ans = 0;
        for (int square = 0; square < 10; square++) 
        {
            ans = (ans + dp(n - 1, square)) % MOD;
        }
        
        return ans;
    }
};
```



# [91. Decode Ways](https://leetcode.com/problems/decode-ways/)

## Solutions

```cpp
class Solution 
{
public:
    int num_decodings(const string& s, int index) 
    {
        if (index == s.length()) 
        {
            return 1;
        }

        if (s[index] == '0') 
        {
            return 0;
        }

        int ways = num_decodings(s, index + 1);

        if (index + 1 < s.length() && stoi(s.substr(index, 2)) <= 26) 
        {
            ways += num_decodings(s, index + 2);
        }

        return ways;
    }

    int numDecodings(string s) 
    {
        return num_decodings(s, 0);
    }
};
```


```cpp
class Solution {
public:
    int numDecodings(string s) 
    {
        int n = s.size();
        if (n == 0 || s[0] == '0') 
        {
            return 0;
        }

        vector<int> dp(n + 1, 0);
        dp[0] = 1;
        dp[1] = 1;

        for (int i = 2; i <= n; ++i) 
        {
            if (s[i - 1] != '0') 
            {
                dp[i] += dp[i - 1];
            }

            int twoDigit = stoi(s.substr(i - 2, 2));
            if (twoDigit >= 10 && twoDigit <= 26) 
            {
                dp[i] += dp[i - 2];
            }
        }

        return dp[n];
    }
};
```

```cpp
// C++ Code
class Solution {
public:
    int numDecodings(string s) 
    {
        int n = s.size();
        if (n == 0 || s[0] == '0') 
        {
            return 0;
        }

        int prev1 = 1; 
        int prev2 = 1; 

        for (int i = 1; i < n; ++i) 
        {
            int current = 0;

            if (s[i] != '0') {
                current += prev1;
            }

            int twoDigit = stoi(s.substr(i - 1, 2));
            if (twoDigit >= 10 && twoDigit <= 26) 
            {
                current += prev2;
            }

            // Update previous states
            prev2 = prev1;
            prev1 = current;
        }

        return prev1;
    }
};
```

## Approach 1: Basic DP with unordered_map

```cpp
class Solution {

public:
    unordered_map<int,int> dp;
   
    int solve(string s,int index)
    {
        if(s[index]=='0')
        {
            return 0;
        }

        if(index>=s.length()-1)
        {
            return 1;
        }

        if(dp.find(index)!=dp.end())
        {
            return dp[index];
        }

        int ways=solve(s,index+1);
        
        if(stoi(s.substr(index,2))<=26)
        {
            ways+=solve(s,index+2);
        }

        
        return dp[index]=ways;
    
    }
    int numDecodings(string s) 
    {
        return solve(s,0);
    }
};
```

## Approach 2: Basic DP with vector

```cpp
class Solution {

public:
    vector<int> dp;
   
    int solve(string s,int index)
    {
        if(s[index]=='0')
        {
            return 0;
        }

        if(index>=s.length()-1)
        {
            return 1;
        }

        if(dp[index]!=-1)
        {
            return dp[index];
        }

        int ways=solve(s,index+1);
        
        if(stoi(s.substr(index,2))<=26)
        {
            ways+=solve(s,index+2);
        }

        dp[index]=ways;
        return dp[index];
    
    }
    int numDecodings(string s) 
    {
        dp.assign(s.length(), -1);
        return solve(s,0);
    }
};
```



## Approach 3: Using Tabulation of data

Example : "1226"

`dp[0]=1`
`dp[1]=1`

Index: 0   1    2    3     4
dp     :1    1    2    3    5

Example : "1026"

`dp[0]=1`
`dp[1]=1`

Index: 0   1    2    3     4
dp     :1    1    1    1    2


Conditions to be Checked

```cpp
			if (s[i - 1] != '0') 
            {
                dp[i] += dp[i-1];
            }

            int twoDigit=stoi(s.substr(i-2, 2));
            if (twoDigit>=10 && twoDigit<=26) 
            {
                dp[i] += dp[i - 2];
            }
```


```cpp
class Solution {
public:
    int numDecodings(string s) 
    {
        int n = s.size();
        if (n == 0 || s[0] == '0') 
        {
            return 0;
        }

        vector<int> dp(n+1, 0);
        dp[0] = 1;
        dp[1] = 1;

        for (int i = 2; i <= n; ++i) 
        {
            if (s[i - 1] != '0') 
            {
                dp[i] += dp[i-1];
            }

            int twoDigit=stoi(s.substr(i-2, 2));
            if (twoDigit>=10 && twoDigit<=26) 
            {
                dp[i] += dp[i - 2];
            }
        }

        return dp[n];
    }
};
```

## Approach 4: Instead of Tabulation save the previous 2 nums

```csharp
class Solution {
public:
    int numDecodings(string s) {
        int n=s.length();
        int next1=1,next2=0;
        for(int i=n-1;i>=0;i--) {
            int curr=next1;
            if( (i+1 < n) && (s.substr(i,2) <= "26") ) curr+=next2;
            if(s[i] == '0') curr=0;

            next2=next1;
            next1=curr;
        }
        return next1;
    }
};
```


# [1155. Number of Dice Rolls With Target Sum](https://leetcode.com/problems/number-of-dice-rolls-with-target-sum/)

## Approach 1: Top Down Approach

```cpp
#define mod 1000000007;
static int fast_io = []() { std::ios::sync_with_stdio(false); std::cin.tie(nullptr); std::cout.tie(nullptr); return 0; }();
class Solution {
public:
    vector<vector<int>> dp;
   
    int numRollsToTarget(int n, int k, int target) 
    {
        dp = vector<vector<int>>(n + 1, vector<int>(target + 1, -1));
        return topDown(n, k, target);
        // return bottomUp(n, k, target);
    }

    int topDown(int diceMoves,int diceFaces,int target)
    {
        if(diceMoves==0)
        {   
            if(target==0)
            {
                return 1;
            }
            return 0;
        }

        if(dp[diceMoves][target]!=-1)
        {
            return dp[diceMoves][target];
        }

        int answer=0;
        
        for(int i=1;i<=diceFaces;i++)
        {
            if((target-i)>=0)
            {
                answer=(answer+topDown(diceMoves-1,diceFaces,target-i))%mod;
            }
        }

        return dp[diceMoves][target] =answer;
    }
};
```

## Approach 2: Bottom Up (TD 💀)

```cpp
class Solution {
public:
    int numRollsToTarget(int d, int f, int target) {
        const int mod = 1000000007;

        // Initialize two vectors to store the current and previous rows of dynamic programming table
        std::vector<int> dp1(target + 1, 0);  // Current row
        std::vector<int> dp2(target + 1, 0);  // Previous row

        // Base case: there is one way to achieve a sum of 0 with 0 dice rolls
        dp1[0] = 1;

        // Dynamic programming loop for each die
        for (int i = 1; i <= d; ++i) {
            int prev = dp1[0];  // Initialize the previous value for the first element
            for (int j = 1; j <= target; ++j) {
                dp2[j] = prev;  // Update dp2 based on the previous value
                prev = (prev + dp1[j]) % mod;  // Update prev for the next iteration

                // If the current sum has more faces than the number of faces on the die, adjust prev
                if (j >= f) prev = (prev - dp1[j - f] + mod) % mod;
            }

            // Swap dp1 and dp2 for the next iteration
            std::swap(dp1, dp2);

            // Reset the first element of dp2 for the next iteration
            dp2[0] = 0;
        }

        // The result is in dp1[target], representing the number of ways to achieve the target sum
        return dp1[target];
    }
};
```

## Approach for space optimization - rolling array

```cpp
class Solution {
public:
    int numRollsToTarget(int d, int f, int target) {
        const int mod = 1000000007;
        std::vector<int> dp(target + 1, 0);
        dp[0] = 1;

        for (int i = 1; i <= d; ++i) {
            std::vector<int> new_dp(target + 1, 0);

            for (int j = 1; j <= target; ++j) {
                for (int k = 1; k <= f && j - k >= 0; ++k) {
                    new_dp[j] = (new_dp[j] + dp[j - k]) % mod;
                }
            }

            std::swap(dp, new_dp);
        }

        return dp[target];
    }
};
```

# [300. Longest Increasing Subsequence 🚀](https://leetcode.com/problems/longest-increasing-subsequence/)

## Prerequisite: What is a subsequence and how to print it.

A **subsequence** is defined as a sequence that can be derived from another string/sequence by deleting some or none of the elements without changing the order of the remaining elements.

*aage se kitne bhi nikal sakte hai peeche se nahi*.

```cpp
// C++ code to print all possible
// subsequences for given array using
// recursion
#include <bits/stdc++.h>
using namespace std;

// Recursive function to print all
// possible subsequences for given array
void printSubsequences(int arr[], int index, vector<int> &subarr,int n)
{
	// Print the subsequence when reach
	// the leaf of recursion tree
	if (index == n)
	{
		for (auto it:subarr)
		{
		    cout << it << " ";
		}
	    if(subarr.size()==0)
		    cout<<"{}";
		    
	    cout<<endl;
	    return;
	 }
	else
	{
		subarr.push_back(arr[index]);
		printSubsequences(arr, index + 1, subarr,n);
		subarr.pop_back();
		printSubsequences(arr, index + 1, subarr,n);
	}
}

int main()
{
	int arr[]={1, 2, 3};
	int n=sizeof(arr)/sizeof(arr[0]);
	vector<int> vec;
	printSubsequences(arr, 0, vec,n);
	return 0;
}

```

## Approach 1: DP

*Time Complexity* : $O(n^2)$
*Space Complexity* : $O(n)$

```cpp
#include <bits/stdc++.h>

using namespace std;

// Function to find the length of the longest increasing subsequence
int getAns(int arr[], int n, int ind, int prev_index, vector<vector<int>>& dp) {

    // Base condition

    if (ind == n)

        return 0;

    if (dp[ind][prev_index + 1] != -1)

        return dp[ind][prev_index + 1];

    int notTake = 0 + getAns(arr, n, ind + 1, prev_index, dp);

    int take = 0;

    if (prev_index == -1 || arr[ind] > arr[prev_index]) {

        take = 1 + getAns(arr, n, ind + 1, ind, dp);

    }

    return dp[ind][prev_index + 1] = max(notTake, take);

}

  

int longestIncreasingSubsequence(int arr[], int n) {

    // Create a 2D DP array initialized to -1

    vector<vector<int>> dp(n, vector<int>(n + 1, -1));

    return getAns(arr, n, 0, -1, dp);

}

  

int main() {

    int arr[] = {10, 9, 2, 5, 3, 7, 101, 18};

    int n = sizeof(arr) / sizeof(arr[0]);

    cout << "The length of the longest increasing subsequence is " << longestIncreasingSubsequence(arr, n);

    return 0;

}

```

## Approach 2: Binary Search

*Time Complexity* : $O(nlogn)$
*Space Complexity* : $O(n)$

```cpp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) 
    {
        int n=nums.size();
        vector<int> temp;
        temp.push_back(nums[0]);
        int len=1;
        for(int i=0;i<n;i++)
        {
            if(nums[i]>temp.back())
            {
                temp.push_back(nums[i]);
                len++;
            }
            else
            {
                int ind=lower_bound(temp.begin(),temp.end(),nums[i])-temp.begin();
                temp[ind]=nums[i];
            }
        }

        return len;
    }
};
```





 😶‍🌫️
# [1235. Maximum Profit in Job Scheduling](https://leetcode.com/problems/maximum-profit-in-job-scheduling/) 😶‍🌫️

## Approach 1: Memoization

*Time Complexity* : $O(n)$
*Space Complexity* : $O(n)$

```cpp
class Solution 
{
int dp[50001];
public:
        int solve(vector<vector<int>>& job, vector<int>& startTime, int n, int ind) 
        {
            if (ind == n) 
            {
                return 0;
            }

            if (dp[ind] != -1) 
            {
                return dp[ind];
            }

            int nextIndex = lower_bound(startTime.begin(),startTime.end(),job[ind][1])-startTime.begin();
            
            int maxProfit = max(solve(job, startTime, n, ind + 1), job[ind][2] + solve(job, startTime, n, nextIndex));

            return dp[ind] = maxProfit;
    }


    int jobScheduling(vector<int>& startTime, vector<int>& endTime, vector<int>& profit) 
    {
        int n = profit.size();
        vector<vector<int>> job;
        memset(dp,-1,sizeof(dp));

        for (int i = 0; i < n; i++) 
        {
            job.push_back({startTime[i],endTime[i],profit[i]});
        }

        sort(job.begin(), job.end());

        for(int i=0;i<n;i++)
        {
            startTime[i]=job[i][0];
        }

        

        return solve(job,startTime,profit.size(),0);
    }
};
```





# [279. Perfect Squares](https://leetcode.com/problems/perfect-squares/)
## Approach 1: Normal recursion with memoization

*Time Complexity* : $O(n)$
*Space Complexity* : $O(n)$

```cpp
class Solution {
public:
    vector<int> memo;  
    int numSquares(int n) {
        memo.resize(n + 1, -1);
        return solve(n);
    }

    int solve(int n)
    {
        if (n == 0) 
        return 0; //end the recursion tree
        if (n < 0) 
        return INT_MAX; //end and to neglect the recusrion tree 
        if (memo[n] != -1) 
        return memo[n]; //find the number in the memo
        int ans = INT_MAX;
        for (int i = 1; i * i <= n; ++i) 
        {
            ans = min(ans, 1 + solve(n - i * i));
        }
        return memo[n]=ans;
    }
};
```


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
``` ```cpp
// C++ program to traverse a map using range 
// based for loop 
#include <bits/stdc++.h> 
using namespace std; 

int main() 
{ 
	int arr[] = { 1, 1, 2, 1, 1, 3, 4, 3 }; 
	int n = sizeof(arr) / sizeof(arr[0]); 
	
	// inserting elements 
	map<int, int> m; 
	for (int i = 0; i < n; i++) 
		m[arr[i]]++; 

	// Printing of MAP 
	cout << "Element Frequency" << endl; 
	for (auto i : m) 
		cout << i.first << " \t\t\t " << i.second << endl; 

	return 0; 
}

```

```cpp
// C++ program to traverse a unordered_map using 
// range based for loop 
#include <bits/stdc++.h> 
using namespace std; 

int main() 
{ 
	int arr[] = { 1, 1, 2, 1, 1, 3, 4, 3 }; 
	int n = sizeof(arr) / sizeof(arr[0]); 

	unordered_map<int, int> m; 
	for (int i = 0; i < n; i++) 
		m[arr[i]]++; 

	// Printing of Unordered_MAP 
	cout << "Element Frequency" << endl; 
	for (auto i : m) 
		cout << i.first << " " << i.second << endl; 

	return 0; 
}

```

Function to find 
```cpp
#include <iostream>
#include <map>

int main() {
    // Create a map
    std::map<int, std::string> myMap;

    // Insert some values into the map
    myMap[1] = "One";
    myMap[2] = "Two";
    myMap[3] = "Three";

    // Find a value in the map
    int keyToFind = 2;
    auto it = myMap.find(keyToFind);

    // Check if the value was found
    if (it != myMap.end()) {
        std::cout << "Value found: " << it->second << std::endl;
    } else {
        std::cout << "Value not found." << std::endl;
    }

    return 0;
}

```
# [Element Appearing More Than 25% In Sorted Array](https://leetcode.com/problems/element-appearing-more-than-25-in-sorted-array/)

## Approach 1: HashMap

*Time Complexity:* :$O(n)$
*Space Complexity*: $O(n)$


```cpp
class Solution {

public:

    int findSpecialInteger(vector<int>& arr)

    {

        int num=arr.size()/4;

        unordered_map<int,int> freq;


        for(int i:arr)
        {
            freq[i]++;
        }

        for(auto i:freq)

        {
            if(i.second>num)
            {
                return i.first;
            }
        }

        return 1;

    }

};
```


## Approach 2: Check at 1/4 distance if same till 3/4 distance

*Time Complexity:* $O(n)$
*Space Complexity*: $O(n)$

```cpp
class Solution {

public:

    int findSpecialInteger(vector<int>& arr)
    {
        int size = arr.size() / 4;
        for (int i = 0; i < arr.size() - size; i++)
        {
            if (arr[i] == arr[i + size])
            {
                return arr[i];
            }
        }
        return -1;
    }

};
```


## Approach 3: Binary Search

*Time complexity: $O(n)$
Space Complexity:* $O(1)$

```cpp
class Solution {

public:

    int findSpecialInteger(vector<int>& arr)

    {

        int n = arr.size();

        vector<int> candidates = {arr[n / 4], arr[n / 2], arr[3 * n / 4]};

        int target = n / 4;

        for (int candidate : candidates)

        {

            int left = lower_bound(arr.begin(), arr.end(), candidate) - arr.begin();

            int right = upper_bound(arr.begin(), arr.end(), candidate) - arr.begin() - 1;

            if (right - left + 1 > target)

            {

                return candidate;

            }

        }

        return -1;

    }

};
```



# [1436. Destination City](https://leetcode.com/problems/destination-city)

## Approach 1: Using Unordered map 


```cpp
class Solution {
public:
    string destCity(vector<vector<string>>& paths) 
    {
        unordered_map<string,string> dest;

        for(int i=0;i<paths.size();i++)
        {
            dest[paths[i][0]]=paths[i][1];
        }

        for(int i=0;i<paths.size();i++)
        {
            auto it=dest.find(paths[i][1]);

            if(it==dest.end())
            {
                return paths[i][1];
            }
        }

        return "";
    }
};
```

## Approach 2: Using Unordered Set

```cpp
class Solution {
public:
    string destCity(vector<vector<string>>& paths) 
    {
        unordered_set<string> hasOutgoing;
        for (int i = 0; i < paths.size(); i++) 
        {
            hasOutgoing.insert(paths[i][0]);
        }
        
        for (int i = 0; i < paths.size(); i++) 
        {
            string candidate = paths[i][1];
            if (hasOutgoing.find(candidate) == hasOutgoing.end()) 
            {
                return candidate;
            }
        }
        
        return "";
    }
};
```





# [441. Arranging Coins](https://leetcode.com/problems/arranging-coins/)

![[Pasted image 20231211165203.jpg]]

![[Pasted image 20231211165226.jpg]]

## Approach 1: Brute Force 

```cpp
class Solution {
public:
    int arrangeCoins(int n) 
    {
       int num=n;
       int cnt=0;
       for(int i=1;i<=n;i++)
       {
           if(num-i>=0)
           {
               num=num-i;
               cnt++;
           }
           else
           {
               return cnt;
           }
       }

    return 1;
    }
};
```

## Approach 2: Math

$k (k+1)/2$ $<=$ $n$
After Solving
$k <= ( \sqrt(2*n) +1/4)-1/2$

```cpp
class Solution {
public:
    int arrangeCoins(int n)
    {
        return (int)(pow((2*(long)n+0.25),0.5)-0.5);
    }
};
```

## Approach 3: Binary Search  *(To Do)*

```cpp
/**
 * Optimized binary search
 *
 * Time Complexity: O(log(N/2)). In case of Int.MAX, time complexity can maximum
 * be O(30) = O(1)
 *
 * Space Complexity: O(1)
 * N = Input number
 */

class Solution {

    public int arrangeCoins(int n) {

        if (n < 0) {

            throw new IllegalArgumentException("Input Number is invalid. Only positive numbers are allowed");

        }

        if (n <= 1) {
            return n;
        }

        if (n <= 3) {
            return n == 3 ? 2 : 1;
        }

        // Binary Search space will start from 2 to n/2.
        long start = 2;
        long end = n / 2;
        while (start <= end) {
            long mid = start + (end - start) / 2;
            long coinsFilled = mid * (mid + 1) / 2;
            if (coinsFilled == n) 
            {
                return (int) mid;
            }

            if (coinsFilled < n) 
            {
                start = mid + 1;
            } 
            else 
            {
                end = mid - 1;
            }

        }

        // Since at this point start > end, start will start pointing to a value greater

        // than the desired result. We will return end as it will point to the correct

        // int value.

        return (int) end;

    }

}
```

^7dd788


# [1716. Calculate Money in Leetcode Bank](https://leetcode.com/problems/calculate-money-in-leetcode-bank/)

## Approach 1: Add 7 on each step

```cpp
class Solution {

public:

    int totalMoney(int n)

    {

        int fullWeeks=n/7;

        int remain=n%7;

        // 1 2 3 4 5 6 7

        // 1 2 3 4 5 6 7 + (7)

        // 1 2 3 4 5 6 7 + (7)*2

        int ans=0;

        if(fullWeeks>0)
	        ans+=28*fullWeeks;
        
        for(int i=1;i<fullWeeks;i++)
        {
            ans+=7*i;
        }
        ans+=(remain*(remain+1))/2 + (remain*fullWeeks);
  
        return ans;

    }

};
```


## Approach 2: Use AP Sum formula

![[Pasted image 20231214125017.jpg]]

```java
class Solution {
    public int totalMoney(int n) {
        int full_weeks=n/7;
        int remain=n%7;
       int a=1,sum=0;
        for(int i=1;i<=full_weeks;i++,a++)
        {
          sum+=((7)*(2*a+6))/2;

        }

        sum+=(remain*(2*a+(remain-1)))/2;

        return sum;
    }
}
```

# [2829. Determine the Minimum Sum of a k-avoiding Array](https://leetcode.com/problems/determine-the-minimum-sum-of-a-k-avoiding-array/)

**Input:** n = 5, k = 4
**Output:** 18
**Explanation:** Consider the k-avoiding array [1,2,4,5,6], which has a sum of 18.
It can be proven that there is no k-avoiding array with a sum less than 18.

## Approach 1: Using a set

```cpp
class Solution 
{
public:
    int minimumSum(int n, int k) 
    {
        set<int> a;
        for(int i=1;i<=n;i++)
        {
            if(a.find(k-i)==a.end())
            {
                a.insert(i);
            }
            else
            {
                n++;
            }
        }

        return accumulate(a.begin(),a.end(),0);
    }
};
```


## Approach 2: Without the set 

```cpp
class Solution {
public:
    int minimumSum(int n, int k) 
    {
        int sum = 0;
        int cnt = 0;

        for(int i=1;i<=n;i++)
        {
            int num=k-i;
            
            //this means that the number is already taken so skip it as it makes k
            if(num>0 && num<i)
            {
                n++; //increase 1 iteration
                continue;
            }

            sum+=i;
           
        }
        return sum;
    }
};
```


# [50. Pow(x, n)](https://leetcode.com/problems/powx-n/)


## Explanation for Approach 2

Certainly! The recursive function is based on the mathematical property:
$$
 x^y = 
\begin{cases} 
1 & \text{if } y = 0 \\
(x^{y/2})^2 & \text{if } y \text{ is even} \\
x \cdot (x^{y/2})^2 & \text{if } y \text{ is odd}
\end{cases}

$$
Let's break down the equations:

1. **Base Case:**
   - If \( y = 0 \), then \( x^0 = 1 \).

2. **Even \( y \):**
   - If \( y \) is even, we can express \( x^y \) as the square of \( x^{y/2} \). This is because \( x^y = (x^{y/2})^2 \).

3. **Odd \( y \):**
   - If \( y \) is odd, we can express \( x^y \) as \( x \) multiplied by the square of \( x^{y/2} \). This accounts for the additional factor of \( x \) that is not covered by the even case. For example, \( x^5 = x \cdot (x^{5/2})^2 \).

In the code:

- The base case checks if \( y \) is 0 and returns 1.
- The recursive step calculates \( x^{y/2} \) and then uses the properties described above to compute \( x^y \) for even and odd \( y \).

So, the code mirrors these mathematical equations to efficiently compute the power of a number.



## Approach 1: Normal brute force

*Time Complexity* : $O(n)$
*Space Complexity* : $O(1)$

```cpp
#include <iostream>

using namespace std;

double myPow(double x, int n) {
    double ans = 1.0;
    for (int i = 0; i < n; i++) {
        ans = ans * x;
    }
    return ans;
}

int main() {
    cout << myPow(2, 10) << endl;
    return 0;
}

```

## Approach 2: Optimized solution

*Time Complexity* : $O(log n)$
*Space Complexity* : $O(log n)$

```cpp
class Solution {
public:
    double myPow(double x, int n) {
        if (n == 0)return 1;
        long long N = n;
        if (n < 0) {
            x = 1 / x;
            N = -N;
        }

    if (N & 1) 
    {
        return x * myPow(x * x, N / 2);
    } 
    else 
    {
        return myPow(x * x, N / 2);
    }
    }
    
};
```

## _This can also be done by divide and conquer but it is not suggest as it has tc and sc as o(n)_









Syntax for SubString:
```cpp
string substr (size_t pos, size_t len) const;
```

```cpp
// C++ program to demonstrate functioning of substr()
#include <iostream>
#include <string>
using namespace std;

int main()
{
	// Take any string
	string s = "dog:cat";

	// Find position of ':' using find()
	int pos = s.find(":");

	// Copy substring after pos
	string sub = s.substr(pos + 1);

	// prints the result
	cout << "String is: " << sub;

	return 0;
}

//Output: String is: cat
```


# Repeated Substring Pattern

## Approach 1:  Compare from the Start and Append

```cpp
class Solution {

public:

    bool repeatedSubstringPattern(string s)

    {

        int n=s.length();

        for(int i=1;i<=n/2;i++)

        {

            string a=s.substr(0,i);

            int nn=a.length();

            if(n%nn==0)

            {

                string comp="";

                for(int i=0;i<n/nn;i++)

                {

                    comp.append(a);

                }

                if(comp==s)

                {

                    return true;

                }

                else

                {

                    continue;

                }

            }

        }

  

    return false;

    }

};
```

## Approach 2: Double remove first and last and then find 

```cpp
class Solution {

public:

    bool repeatedSubstringPattern(string s)

    {

        string doubled = s + s;

        string sub = doubled.substr(1, doubled.size() - 2);

        return sub.find(s) != string::npos;

    }

};
```


## Approach 3:  Start from middle and increase the count 

```cpp
class Solution {

public:

    bool repeatedSubstringPattern(string s) {

        int n=s.size();

        for(int i=n/2;i>=1;i--)

        {              

            cout<<s.substr(0,n-i)<<" "<<s.substr(i)<<i<<"\n";

            if(n%i==0)

            {                                                          

                if(s.substr(0,n-i)==s.substr(i))

                {

                    cout<<s.substr(0,n-i)<<" "<<s.substr(i)<<i<<"\n";

                    return true;          

                }

            }

        }

        return false;

    }

};
```






# [242. Valid Anagram](https://leetcode.com/problems/valid-anagram/)

```cpp
class Solution {
public:
    bool isAnagram(string s, string t) 
    {
	    if(s.length()!=t.length())
        {
            return false;
        }
        vector<int> freq(26,0);

        for(int i=0;i<s.size();i++)
        {
            int a=s[i]-'a';
            freq[a]++;
        }

        for(int i=0;i<t.size();i++)
        {
            int a=t[i]-'a';
            if(freq[a]>0)
            {
                freq[a]--;
                continue;
            }
            else
            {
                return false;
            }
        }

        for(int i=0;i<freq.size();i++)
        {
            if(freq[i]!=0)
            {
                return false;
            }
        }

        
        return true;
    }
};
```

```cpp
class Solution {
public:
    bool isAnagram(string s, string t) 
    {
        sort(s.begin(),s.end());
        sort(t.begin(),t.end());
        return s==t;
    }
};
```



# [1422. Maximum Score After Splitting a String](https://leetcode.com/problems/maximum-score-after-splitting-a-string/)

## Approach 1: Brute Force

*Time Complexity* : $O(n^2)$
*Space Complexity* : $O(1)$

```cpp
static int fast_io = []() { std::ios::sync_with_stdio(false); std::cin.tie(nullptr); std::cout.tie(nullptr); return 0; }();
class Solution {
public:
    int maxScore(string s) 
    {
        int n=s.length();
        int score=INT_MIN;

        for(int i=0;i<n-1;i++)
        {
            int num=0;
            for(int j=0;j<=i;j++)
            {
                if(s[j]=='0')
                {
                    num++;
                }
            }

            for(int j=i+1;j<n;j++)
            {
                if(s[j]=='1')
                {
                    num++;
                }
            }

            score=max(num,score);
        }
        return score;
    }
};
```

## Approach 2: 

*Time Complexity* : $O(2*n)$
*Space Complexity* : $O(1)$


Count function:
`int ones=count(s.begin(),s.end(),'1');`


```cpp
class Solution {
public:
    int maxScore(string s) 
    {
        int ones=count(s.begin(),s.end(),'1');

        int score=0;
        int zeros=0;

        for(int i=0;i<s.size()-1;i++)
        {
            if(s[i]=='1')
            {
                ones--;
            }
            else
            {
                zeros++;
            }

            score = max(score, zeros+ones);
        }

        return score;
    }
};
```







# [1496. Path Crossing](https://leetcode.com/problems/path-crossing/)

## Approach 1: Push Coordinates in the Visited vector and find

*Time Complexity* : $O(n)$
*Space Complexity* : $O(n)$

```cpp
class Solution {
public:
    bool isPathCrossing(string path) 
    {
        vector<int> point(2,0);
        vector<vector<int>> visited;
        visited.push_back({0,0});
        for(char i:path)
        {
            if(i=='N')
            {
                point[1]++;
            }
            else if(i=='S')
            {
                point[1]--;
            }
            else if(i=='E')
            {
                point[0]++;
            }
            else if(i=='W')
            {
                point[0]--;
            }
            cout<<point[0]<<" "<<point[1]<<"\n";
            if(find(visited.begin(),visited.end(),point)==visited.end())
            {
                visited.push_back(point);
            }
            else
            {
                return true;
            }
        }

        return false;
    }
};
```

## Approach 2: Using Map

*Time Complexity* : $O(n)$
*Space Complexity* : $O(n)$

```cpp
class Solution {
public:
    bool isPathCrossing(string path) {
        unordered_map<char, pair<int, int>> moves;
        moves['N'] = {0, 1};
        moves['S'] = {0, -1};
        moves['W'] = {-1, 0};
        moves['E'] = {1, 0};
        
        unordered_set<string> visited;
        visited.insert("0,0");
        
        int x = 0;
        int y = 0;
        
        for (char c : path) {
            pair<int, int> curr = moves[c];
            int dx = curr.first;
            int dy = curr.second;
            x += dx;
            y += dy;
            
            string hash = to_string(x) + "," + to_string(y);
            if (visited.find(hash) != visited.end()) 
            {
                return true;
            }
            
            visited.insert(hash);
        }
        
        return false;
    }
};
```






# [91. Decode Ways](https://leetcode.com/problems/decode-ways/)

## Solutions

```cpp
class Solution 
{
public:
    int num_decodings(const string& s, int index) 
    {
        if (index == s.length()) 
        {
            return 1;
        }

        if (s[index] == '0') 
        {
            return 0;
        }

        int ways = num_decodings(s, index + 1);

        if (index + 1 < s.length() && stoi(s.substr(index, 2)) <= 26) 
        {
            ways += num_decodings(s, index + 2);
        }

        return ways;
    }

    int numDecodings(string s) 
    {
        return num_decodings(s, 0);
    }
};
```


```cpp
class Solution {
public:
    int numDecodings(string s) 
    {
        int n = s.size();
        if (n == 0 || s[0] == '0') 
        {
            return 0;
        }

        vector<int> dp(n + 1, 0);
        dp[0] = 1;
        dp[1] = 1;

        for (int i = 2; i <= n; ++i) 
        {
            if (s[i - 1] != '0') 
            {
                dp[i] += dp[i - 1];
            }

            int twoDigit = stoi(s.substr(i - 2, 2));
            if (twoDigit >= 10 && twoDigit <= 26) 
            {
                dp[i] += dp[i - 2];
            }
        }

        return dp[n];
    }
};
```

```cpp
// C++ Code
class Solution {
public:
    int numDecodings(string s) 
    {
        int n = s.size();
        if (n == 0 || s[0] == '0') 
        {
            return 0;
        }

        int prev1 = 1; 
        int prev2 = 1; 

        for (int i = 1; i < n; ++i) 
        {
            int current = 0;

            if (s[i] != '0') {
                current += prev1;
            }

            int twoDigit = stoi(s.substr(i - 1, 2));
            if (twoDigit >= 10 && twoDigit <= 26) 
            {
                current += prev2;
            }

            // Update previous states
            prev2 = prev1;
            prev1 = current;
        }

        return prev1;
    }
};
```

## Approach 1: Basic DP with unordered_map

```cpp
class Solution {

public:
    unordered_map<int,int> dp;
   
    int solve(string s,int index)
    {
        if(s[index]=='0')
        {
            return 0;
        }

        if(index>=s.length()-1)
        {
            return 1;
        }

        if(dp.find(index)!=dp.end())
        {
            return dp[index];
        }

        int ways=solve(s,index+1);
        
        if(stoi(s.substr(index,2))<=26)
        {
            ways+=solve(s,index+2);
        }

        
        return dp[index]=ways;
    
    }
    int numDecodings(string s) 
    {
        return solve(s,0);
    }
};
```

## Approach 2: Basic DP with vector

```cpp
class Solution {

public:
    vector<int> dp;
   
    int solve(string s,int index)
    {
        if(s[index]=='0')
        {
            return 0;
        }

        if(index>=s.length()-1)
        {
            return 1;
        }

        if(dp[index]!=-1)
        {
            return dp[index];
        }

        int ways=solve(s,index+1);
        
        if(stoi(s.substr(index,2))<=26)
        {
            ways+=solve(s,index+2);
        }

        dp[index]=ways;
        return dp[index];
    
    }
    int numDecodings(string s) 
    {
        dp.assign(s.length(), -1);
        return solve(s,0);
    }
};
```



## Approach 3: Using Tabulation of data

Example : "1226"

`dp[0]=1`
`dp[1]=1`

Index: 0   1    2    3     4
dp     :1    1    2    3    5

Example : "1026"

`dp[0]=1`
`dp[1]=1`

Index: 0   1    2    3     4
dp     :1    1    1    1    2


Conditions to be Checked

```cpp
			if (s[i - 1] != '0') 
            {
                dp[i] += dp[i-1];
            }

            int twoDigit=stoi(s.substr(i-2, 2));
            if (twoDigit>=10 && twoDigit<=26) 
            {
                dp[i] += dp[i - 2];
            }
```


```cpp
class Solution {
public:
    int numDecodings(string s) 
    {
        int n = s.size();
        if (n == 0 || s[0] == '0') 
        {
            return 0;
        }

        vector<int> dp(n+1, 0);
        dp[0] = 1;
        dp[1] = 1;

        for (int i = 2; i <= n; ++i) 
        {
            if (s[i - 1] != '0') 
            {
                dp[i] += dp[i-1];
            }

            int twoDigit=stoi(s.substr(i-2, 2));
            if (twoDigit>=10 && twoDigit<=26) 
            {
                dp[i] += dp[i - 2];
            }
        }

        return dp[n];
    }
};
```

## Approach 4: Instead of Tabulation save the previous 2 nums

```cpp
class Solution {
public:
    int numDecodings(string s) {
        int n=s.length();
        int next1=1,next2=0;
        for(int i=n-1;i>=0;i--) {
            int curr=next1;
            if( (i+1 < n) && (s.substr(i,2) <= "26") ) curr+=next2;
            if(s[i] == '0') curr=0;

            next2=next1;
            next1=curr;
        }
        return next1;
    }
};
```A spell to make the code run faster
```cpp
static int fast_io = []() { std::ios::sync_with_stdio(false); std::cin.tie(nullptr); std::cout.tie(nullptr); return 0; }();
```

```cpp
ios_base::sync_with_stdio(0),cin.tie(0),cout.tie(0);
```

```cpp
#pragma GCC optimize("O3", "unroll-loops");
```

```cpp
Solution() { ios_base::sync_with_stdio(false); cin.tie(NULL); }
```- [ ] Fenwick Tree
- [ ] Syntax:
```cpp
std::vector<int> myVector;  // Creates an empty vector of integers
```

```cpp
std::vector<int> myVector(5);  // Creates a vector with 5 default-initialized elements (0 for int)
```

```cpp
std::vector<int> myVector(5, 42);  // Creates a vector with 5 elements, each initialized to 42
```

```cpp
int arr[] = {1, 2, 3, 4, 5};
std::vector<int> myVector(arr, arr + sizeof(arr) / sizeof(arr[0]));
```

```cpp
std::vector<int> myVector = {1, 2, 3, 4, 5};  // Creates a vector with specified elements
```

```cpp
std::vector<int> sourceVector = {1, 2, 3, 4, 5};
std::vector<int> myVector(sourceVector);  // Creates a new vector as a copy of another vector
```

```cpp
std::vector<int> sourceVector = {1, 2, 3, 4, 5};
std::vector<int> myVector(sourceVector.begin(), sourceVector.end());
```

```cpp
int row[n] = {0};
```
2D Vector
```cpp
vector<vector<int>> res(matrix[0].size(), vector<int>(matrix.size()));
vector<vector<int>> res(columns, vector<int>(rows));
```

Code for find in the vector
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    int ser = 3;

    auto it = std::find(vec.begin(), vec.end(), ser);

    if (it != vec.end()) {
        std::cout << "Element " << ser << " found at position: ";
        std::cout << std::distance(vec.begin(), it) << " (counting from zero)\n";
    }

    return 0;
}
```

Used to assign parameter to global vector
```cpp
dp.assign(s.length(), -1);
```

![[Pasted image 20231210115640.png]]

Accumulate function used to take sum of all the numbers of the array:

```cpp
 int rightSum=accumulate(nums.begin(), nums.end(), 0);
```



# [118. Pascal's Triangle](https://leetcode.com/problems/pascals-triangle/)


^b36dfa

![[Pasted image 20231210120727.png]]

## Approach 1: Using Combinatorial Formula

*Time Complexity*: $O(n^2)$


```cpp
class Solution {

public:

    vector<vector<int>> generate(int numRows)

    {

        vector<vector<int>> pascal;

  

        for(int i=0;i<numRows;i++)

        {

            vector<int> temp(i+1,1);

            for(int j=1;j<i;j++)

            {

                temp[j]=pascal[i-1][j-1]+pascal[i-1][j];

            }

            pascal.push_back(temp);

        }

  

        return pascal;

    }

};
```

## Approach 2: Recursion

*Time Complexity*: $O(n^2)$

```cpp
class Solution {

public:

    vector<vector<int>> generate(int numRows)

    {

        if(numRows==0)

        return {};

  

        if(numRows==1)

        return {{1}};

  

        vector<vector<int>> pascal=generate(numRows-1);

        vector<int> newRow(numRows,1);

  

        for(int i=1;i<numRows-1;i++)

        {

            newRow[i]=pascal.back()[i-1]+pascal.back()[i];

        }

  

        pascal.push_back(newRow);

        return pascal;

    }

};
```

## Approach 3: Dynamic Programming

*Time Complexity*: $O(n)$

The time complexity is less because the we use the preRow to make the current row making it time efficient than previous algos 

```cpp
class Solution {

public:

    vector<vector<int>> generate(int numRows)

    {

        vector<vector<int>> pascal;

        vector<int> preRow;

  

        for(int i=0;i<numRows;i++)

        {

            vector<int> current(i+1,1);

            for(int j=1;j<i;j++)

            {

                current[j]=preRow[j-1]+preRow[j];

            }

  

            pascal.push_back(current);

            preRow=current;

        }

  

        return pascal;

    }

};
```



# [121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) 🚀

^68a8cf

![[Pasted image 20231211154909.png]]

## Approach 1: 

1. Set the profit as 0 and minimum price as max number
2. Find the Minimum price `min_price` at every step
3. at the exact price find if there is a more profit than 0 at that step
	1. If yes update the profit 
	2. else continue 
4. Update

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) 
    {
        int sell=0;
        int min_price=INT_MAX;
        for(int i:prices)
        {
            min_price=min(min_price,i);
            sell=max(sell,i-min_price);
        }

        return sell;
    }
};
```

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) 
    {
        int min_price=INT_MAX;
        int profit=0;
        for(int i=0;i<prices.size();i++)
        {
            if(min_price>prices[i])
            {
                min_price=prices[i];
            }
            if(prices[i]-min_price>profit)
            {
                profit=prices[i]-min_price;
            }
        }

        return profit;
    }
};
```

```cpp

#include<bits/stdc++.h>
using namespace std;

int maxProfit(vector<int> &arr) {
    int maxPro = 0;
    int n = arr.size();

    for (int i = 0; i < n; i++) 
    {
        for (int j = i + 1; j < n; j++) 
        {
            if (arr[j] > arr[i]) 
            {
	            maxPro = max(arr[j] - arr[i], maxPro);
            }
        }
    }

    return maxPro;
}

int main() {
    vector<int> arr = {7,1,5,3,6,4};
    int maxPro = maxProfit(arr);
    cout << "Max profit is: " << maxPro << endl;
}

```
# [1685. Sum of Absolute Differences in Sorted Array](https://leetcode.com/problems/sum-of-absolute-differences-in-a-sorted-array/)

^430655

---
**Input:** nums = [2,3,5]
**Output:** [4,3,5]
**Explanation:** Assuming the arrays are 0-indexed, then
result[0] = |2-2| + |2-3| + |2-5| = 0 + 1 + 3 = 4,
result[1] = |3-2| + |3-3| + |3-5| = 1 + 0 + 2 = 3,
result[2] = |5-2| + |5-3| + |5-5| = 3 + 2 + 0 = 5.

---
$Absolute Sum = max(a,b)-min(a,b)$

## Approach 1: Calculate Prefix Sum of Right and Left

1. Keep left and right sum , and then do further calculating 
2.  Used formula for this is:
	` ans[i]=rightSum-(nums[i]*rightN)-nums[i]+(nums[i]*leftN)-leftSum;`

*Time Complexity:* :$O(n)$
*Space Complexity*: $O(1)$

---

```cpp
class Solution {

public:

    vector<int> getSumAbsoluteDifferences(vector<int>& nums)

    {

        vector<int> ans(nums.size(),0);

       int rightSum=0;

       for(int i:nums)

       {
           rightSum+=i;
       }

  
       int leftSum=0;
       int leftN=0;
       int rightN=nums.size()-1;
  
       for(int i=0;i<nums.size();i++)
       {
           ans[i]=rightSum-(nums[i]*rightN)-nums[i]+(nums[i]*leftN)-leftSum;
           leftN++;
           rightN--;
           leftSum+=nums[i];
           rightSum-=nums[i];
       }

    return ans;
    }

};
```

## Approach 2:  Cumulative array of the nums array

1. Make a cumulative sum array and calculate right sum and left sum of the number on each iteration like the previous approach.

```cpp
class Solution {
public:
    vector<int> getSumAbsoluteDifferences(vector<int>& nums) {
        int n = nums.size();
        vector<int> prefix = {nums[0]};
        for (int i = 1; i < n; i++) {
            prefix.push_back(prefix[i - 1] + nums[i]);
        }
        
        vector<int> ans;
        for (int i = 0; i < n; i++) 
        {
            int leftSum = prefix[i] - nums[i];
            int rightSum = prefix[n - 1] - prefix[i];
            
            int leftCount = i;
            int rightCount = n - 1 - i;
            
            int leftTotal = leftCount * nums[i] - leftSum;
            int rightTotal = rightSum - rightCount * nums[i];
            
            ans.push_back(leftTotal + rightTotal);
        }
        
        return ans;
    }
};
```

# [2482. Difference Between Ones and Zeros in Row and Column](https://leetcode.com/problems/difference-between-ones-and-zeros-in-row-and-column/)

^2205d5


## Approach 1: Count number of Zeros in columns and rows 


```cpp
class Solution {

public:

    vector<vector<int>> onesMinusZeros(vector<vector<int>>& grid)

    {

        vector<int> rowZero(grid.size(),0);

        vector<int> colZero(grid[0].size(),0);

  

        for(int i=0;i<grid.size();i++)

        {

            for(int j=0;j<grid[0].size();j++)

            {

                if(grid[i][j]==0)

                {
                    rowZero[i]++;
                    colZero[j]++;
                }

            }

        }

  

        vector<vector<int>> ans(grid.size(), vector<int>(grid[0].size(),0));

  

        for(int i=0;i<grid.size();i++)

        {

            for(int j=0;j<grid[0].size();j++)

            {

                ans[i][j]=(grid[0].size()-rowZero[i])+(grid.size()-colZero[j])- rowZero[i] - colZero[j];

            }

        }

  

        return ans;

    }

};
```



# [73. Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/) 🚀

## Approach 1: Using an function and extra space

**Time Complexity:** $O((N*M)*(N + M)) + O(N*M)$


```cpp
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) 
    {
        ios_base::sync_with_stdio(0), cin.tie(0), cout.tie(0);
        int n=matrix.size();
        int m=matrix[0].size();
        vector<vector<int>> matrix1(matrix);

        for(int i=0;i<n;i++)
        {
            for(int j=0;j<m;j++)
            {
                if(matrix1[i][j]==0)
                {
                    setRowCol(i,j,matrix);
                }
            }
        }
    }

    void setRowCol(int row, int col,vector<vector<int>>& matrix)
    {
        int rows = matrix.size();
        int columns = matrix[0].size();

        for(int i=0;i<columns;i++)
        {
            matrix[row][i]=0;
        }
        for(int i=0;i<rows;i++)
        {
            matrix[i][col]=0;
        }
        
    }
};
```

## Approach 2: Marking -1 and then replacing it to 0

```cpp

#include <bits/stdc++.h>
using namespace std;

void markRow(vector<vector<int>> &matrix, int n, int m, int i) {
    // set all non-zero elements as -1 in the row i:
    for (int j = 0; j < m; j++) {
        if (matrix[i][j] != 0) {
            matrix[i][j] = -1;
        }
    }
}


void markCol(vector<vector<int>> &matrix, int n, int m, int j) {
    // set all non-zero elements as -1 in the col j:
    for (int i = 0; i < n; i++) {
        if (matrix[i][j] != 0) {
            matrix[i][j] = -1;
        }
    }
}

vector<vector<int>> zeroMatrix(vector<vector<int>> &matrix, int n, int m) {

    // Set -1 for rows and cols
    // that contains 0. Don't mark any 0 as -1:

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (matrix[i][j] == 0) {
                markRow(matrix, n, m, i);
                markCol(matrix, n, m, j);
            }
        }
    }

    // Finally, mark all -1 as 0:
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (matrix[i][j] == -1) {
                matrix[i][j] = 0;
            }
        }
    }

    return matrix;
}

int main()
{
    vector<vector<int>> matrix = {{1, 1, 1}, {1, 0, 1}, {1, 1, 1}};
    int n = matrix.size();
    int m = matrix[0].size();
    vector<vector<int>> ans = zeroMatrix(matrix, n, m);

    cout << "The Final matrix is: n";
    for (auto it : ans) {
        for (auto ele : it) {
            cout << ele << " ";
        }
        cout << "n";
    }
    return 0;
}

```

## Approach 3: Taking 2 matrix for rows and columns and marking them

```cpp
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) 
    {
        int n=matrix.size();
        int m=matrix[0].size();
        vector<int> row(n,0);
        vector<int> col(m,0);

        for (int i=0; i<matrix.size(); i++) 
        {
            for (int j=0; j<matrix[0].size(); j++) 
            {
                if (matrix[i][j] == 0) 
                {
                    row[i] = 1;
                    col[j] = 1;
                }
            }
        }

         for (int i=0; i<matrix.size(); i++) 
        {
            for (int j=0; j<matrix[0].size(); j++) 
            {
                if(row[i]==1 || col[j]==1)
                {
                    matrix[i][j]=0;
                }
            }
        }

    }
};
```


## Approach 4: Optimized by using the 1st col and row as matrix


```cpp
class Solution {

public:

    void setZeroes(vector<vector<int>>& matrix)

    {

        int col0=1;

  

        for(int i=0;i<matrix.size();i++)

        {

            for(int j=0;j<matrix[0].size();j++)

            {

                if(matrix[i][j]==0)

                {

                    matrix[i][0]=0;

  

                    if(j==0)

                    {

                        col0=0;

                    }

                    else

                    {

                        matrix[0][j]=0;

                    }

                }

            }

        }

  

            for(int i=1;i<matrix.size();i++)

            {

                for(int j=1;j<matrix[0].size();j++)

                {

                    if (matrix[i][j] != 0)

                    {

                        if (matrix[i][0] == 0 || matrix[0][j] == 0)

                        {

                            matrix[i][j] = 0;

                        }

                    }

                }

            }

  

            if(matrix[0][0]==0)

            {

                for(int i=0;i<matrix[0].size();i++)

                {

                    matrix[0][i]=0;

                }

            }

  

            if(col0==0)

            {

                for(int i=0;i<matrix.size();i++)

                {

                    matrix[i][0]=0;

                }

            }

    }

};
```

# [31. Next Permutation](https://leetcode.com/problems/next-permutation/) 🚀

##  Approach 1: Brute Force , Get all the permutation of the number then get the next number 

## Approach 2: Using inbuilt function

```cpp
class Solution {
public:
    void nextPermutation(vector<int>& nums) 
    {
        next_permutation(nums.begin(),nums.end());
    }
};
```

## Approach 3: Optimized Solution

```cpp
class Solution 
{
public:
    void nextPermutation(std::vector<int>& nums) 
    {
        int n = nums.size();
        int index = -1;

        for (int i = n - 1; i > 0; i--) 
        {
            if (nums[i - 1] < nums[i]) 
            {
                index = i - 1;
                break;
            }
        }

        if (index == -1) 
        {
            std::reverse(nums.begin(), nums.end());
            return;
        }

        int smallestGreaterIndex = index + 1;
        for (int i = index + 1; i < n; i++) 
        {
            if (nums[i] > nums[index] && nums[i] <= nums[smallestGreaterIndex]) 
            {
                smallestGreaterIndex = i;
            }
        }

        swap(nums[index], nums[smallestGreaterIndex]);

        reverse(nums.begin() + index + 1, nums.end());
    }
};
```


# [75. Sort Colors](https://leetcode.com/problems/sort-colors/)

## Approach 1 : Using Hash Map

*Time Complexity:* :$O(2n)$
*Space Complexity*: $O(1)$


```cpp
class Solution {
public:
    void sortColors(std::vector<int>& nums) 
    {
        std::vector<int> freq(3, 0);

        for (int i : nums)
        {
            freq[i]++;
        }

        int index = 0; 
        for (int i = 0; i < 3; i++)
        {
            int numF = freq[i];
            for (int j = 0; j < numF; j++)
            {
                nums[index] = i;
                index++; 
            }
        }
    }
};
```

## Approach 2: Dutch National Flag Algorithm

*Time Complexity:* :$O(n)$
*Space Complexity*: $O(1)$

![[Pasted image 20231214193127.png]]

While mid < high
and if mid exceeds the high than whole array is sorted

![[Pasted image 20231214194508.png]]

```cpp
class Solution {
public:
    void sortColors(std::vector<int>& nums) 
    {
        int low=0;
        int high=nums.size()-1;
        int mid=0;
        
        while(mid<=high)
        {
            if(nums[mid]==0)
            {
                swap(nums[low],nums[mid]);
                low++;
                mid++;
            }
            else if(nums[mid]==1)
            {
                mid++;
            }
            else
            {
                swap(nums[mid],nums[high]);
                high--;
            }
        }
    }
};
```

# [53. Maximum Subarray / Kadane's Algo](https://leetcode.com/problems/maximum-subarray/) 🚀

Here it continuosly added makes the current sum the addition of number if it is greather than the number to be added

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) 
    {
        int maxSum = nums[0];
        int currentSum = nums[0];

        for (int i = 1; i < nums.size(); i++) 
        {
            currentSum = max(nums[i], currentSum + nums[i]);
            maxSum = max(maxSum, currentSum);
        }

        return maxSum;
    }
};
```


There are two counters;
sum and maxSum
We continuously add the number to the sum and in the counter sum we continuously update the maxSum.

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) 
    {
    int maxi = INT_MIN;
    long long sum = 0;

    for (int i = 0; i < nums.size(); i++) 
    {

        sum += nums[i];

        if (sum > maxi) 
        {
            maxi = sum;
        }

        if (sum < 0) 
        {
            sum = 0;
        }
    }

    return maxi;
    }
```


# [48. Rotate Image](https://leetcode.com/problems/rotate-image/) 🚀

```cpp
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) 
    {
        int n=matrix.size();
        vector<vector<int>> res(matrix);

        for(int i=0;i<matrix.size();i++)
        {
            for(int j=0;j<matrix[0].size();j++)
            {
                res[j][n-1-i]=matrix[i][j];
            }
        }

        matrix=res;
    }
};
```

```cpp
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) 
    {
        for(int i=0;i<matrix.size();i++)
        {
            for(int j=0;j<=i;j++)
            {
              swap(matrix[i][j], matrix[j][i]);
            }
        }

        for(int i=0;i<matrix.size();i++)
        {
            reverse(matrix[i].begin(), matrix[i].end());
        }

    }
};
```

# [46. Permutations](https://leetcode.com/problems/permutations/)

```cpp
class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) 
    {
        vector<vector<int>> ans;

        permuteRecursive(nums, 0, ans);
	    return ans;
    }

    void permuteRecursive(vector<int> &num,int begin, vector<vector<int>> &result)
    {
        if(begin==num.size())
        {
            result.push_back(num);
            return;
        }

        for(int i=begin;i<num.size();i++)
        {
            swap(num[begin],num[i]);
            permuteRecursive(num, begin + 1, result);
		    swap(num[begin], num[i]);
        }
    }
};
```


# [2391. Minimum Amount of Time to Collect Garbage](https://leetcode.com/problems/minimum-amount-of-time-to-collect-garbage/)

## Approach 1 (Mine): Find the position of where truck finishes it work

```cpp
static int fast_io = []() { std::ios::sync_with_stdio(false); std::cin.tie(nullptr); std::cout.tie(nullptr); return 0; }();
class Solution {
public:
    int garbageCollection(vector<string>& garbage, vector<int>& travel) 
    {
        int G_last=0;
        int M_last=0;
        int P_last=0;
        int c=0;
        int cur=0;

        for(int i=1;i<garbage.size();i++)
        {   
             cur+=travel[i-1];
            for(char a:garbage[i])
            {
                c++;
                if(a=='G')
                {
                    G_last=cur;
                }
                else if(a=='P')
                {
                    P_last=cur;
                }
                else if(a=='M')
                {
                    M_last=cur;
                }
            }
        }

        return G_last+M_last+P_last+c+garbage[0].length();
    }
};
```

## Approach 2: HashMap & Prefix Sum 

As such hashmap is not required in this sum it can be done without hashmap

```cpp
class Solution {
public:
    int garbageCollection(vector<string>& garbage, vector<int>& travel) 
    {
        // Vector to store the prefix sum in travel.
        vector<int> prefixSum(travel.size() + 1, 0);
        prefixSum[1] = travel[0];
        for (int i = 1; i < travel.size(); i++) 
        {
            prefixSum[i + 1] = prefixSum[i] + travel[i];
        }
        
        // Map to store garbage type to the last house index.
        unordered_map<char, int> garbageLastPos;
        unordered_map<char, int> garbageCount;
        
        for (int i = 0; i < garbage.size(); i++) 
        {
            for (char c : garbage[i]) 
            {
                garbageLastPos[c] = i;
                garbageCount[c]++;
            }
        }
        
        char garbageTypes[3] = {'M', 'P', 'G'};
        int ans = 0;
        for (char c : garbageTypes) 
        {
            // Add only if there is at least one unit of this garbage.
            if (garbageCount[c]) 
            {
                ans += prefixSum[garbageLastPos[c]] + garbageCount[c];
            }
        }
        
        return ans;
    }
};
```



# [268. Missing Number](https://leetcode.com/problems/missing-number/)

## Approach 1: Assuming Extra number than subtracting 

**Input:** nums =$[3,0,1]$
**Output:** 2

We calculate the sum of $[0,1,2,3]$ 

```cpp
class Solution {
public:
    int missingNumber(vector<int>& nums) 
    {   
        int n=nums.size();
        int sum=0;
        int required=(n*(n+1))/2;
        for(int i=0;i<n;i++)
        {
            sum+=nums[i];
        }


    return required-sum;
    }
};
```

## Approach 2: Using XOR

from above example 
 $[3,0,1]$ 

nums = 3^1
num=3^1^0^2^1^3

leaving - 0^2 = 2-----Ans

```cpp
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int num=nums[0]^1;
        for(int i=1;i<nums.size();i++){
            num = num^nums[i]^(i+1);
        }
        return num;
    }
};
```





# [1913. Maximum Product Difference Between Two Pairs](https://leetcode.com/problems/maximum-product-difference-between-two-pairs/)

Find the Second Largest and Smallest Number

Continuously Replace the Largest and put the previous large in the Second Large Number

```cpp
class Solution {
public:
    int maxProductDifference(vector<int>& nums) {

        int smallest=INT_MAX;
        int secondSmallest=INT_MAX;
        int biggest=0;
        int secondBiggest=0;
        for(int i=0;i<nums.size();i++)
        {
            if(nums[i]>biggest)
            {
                secondBiggest=biggest;
                biggest=nums[i];
            }
            else
            {
                secondBiggest=max(nums[i],secondBiggest);
            }
            if(nums[i]<smallest)
            {
                secondSmallest=smallest;
                smallest=nums[i];
            }
            else
            {
                secondSmallest=min(nums[i],secondSmallest);
            }
        }
        return ((biggest*secondBiggest)-(smallest*secondSmallest));
    }
};
```


# [56. Merge Intervals](https://leetcode.com/problems/merge-intervals/) 🚀

This can also be done using graphs using making graphs and doing Dfs over it 


used `ans.back()` which give the last element of the vector

## Approach 1

```cpp
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) 
    {
        sort(intervals.begin(),intervals.end());
        vector<vector<int>> ans;
        int n=intervals.size();

        for(int i=0;i<n;i++)
        {
            if(ans.empty() || ans.back()[1]<intervals[i][0])
            {
                ans.push_back(intervals[i]);
            }
            else
            {
                ans.back()[1]=max(intervals[i][1],ans.back()[1]);
            }
        }

        return ans;
    }
};
```

## Approach 2 : Using graph (TD💀)

```cpp
class Solution {
public:
    map<vector<int>, vector<vector<int>>> graph;
    map<int, vector<vector<int>>> nodes_in_comp;
    set<vector<int>> visited;

    bool overlap(vector<int>& a, vector<int>& b) {
        return a[0] <= b[1] and b[0] <= a[1];
    }

    // build a graph where an undirected edge between intervals u and v exists
    // iff u and v overlap.
    void buildGraph(vector<vector<int>>& intervals) {
        for (auto interval1 : intervals) {
            for (auto interval2 : intervals) {
                if (overlap(interval1, interval2)) {
                    graph[interval1].push_back(interval2);
                    graph[interval2].push_back(interval1);
                }
            }
        }
    }

    // merges all of the nodes in this connected component into one interval.
    vector<int> mergeNodes(vector<vector<int>>& nodes) {
        int min_start = nodes[0][0];
        for (auto node : nodes) {
            min_start = min(min_start, node[0]);
        }

        int max_end = nodes[0][1];
        for (auto node : nodes) {
            max_end = max(max_end, node[1]);
        }

        return {min_start, max_end};
    }

    // use depth-first search to mark all nodes in the same connected component
    // with the same integer.
    void markComponentDFS(vector<int>& start, int comp_number) {
        stack<vector<int>> stk;
        stk.push(start);

        while (!stk.empty()) {
            vector<int> node = stk.top();
            stk.pop();

            // not found
            if (visited.find(node) == visited.end()) {
                visited.insert(node);

                nodes_in_comp[comp_number].push_back(node);

                for (auto child : graph[node]) {
                    stk.push(child);
                }
            }
        }
    }

    // gets the connected components of the interval overlap graph.
    void buildComponents(vector<vector<int>>& intervals) {
        int comp_number = 0;

        for (auto interval : intervals) {
            if (visited.find(interval) == visited.end()) {
                markComponentDFS(interval, comp_number);
                comp_number++;
            }
        }
    }

    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        buildGraph(intervals);
        buildComponents(intervals);

        // for each component, merge all intervals into one interval.
        vector<vector<int>> merged;
        for (size_t comp = 0; comp < nodes_in_comp.size(); comp++) {
            merged.push_back(mergeNodes(nodes_in_comp[comp]));
        }

        return merged;
    }
};
```



# [287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)🚀

## Approach 1: Using Hash Table

```cpp
class Solution {
public:
    int findDuplicate(vector<int>& nums) 
    {
        int n=nums.size();
        int maxy=INT_MIN;
        for(int i=0;i<n;i++)
        {
            maxy=max(maxy,nums[i]);
        }

        vector<int> freq(maxy+1,0);

        for(int i=0;i<n;i++)
        {
            freq[nums[i]]++;
            if(freq[nums[i]]==2)
            {
                return nums[i];
            }
        }

        return -1;
    }
};
```

## Approach 2 : Sort and Find 

```cpp
class Solution {
public:
    int findDuplicate(vector<int>& nums) 
    {
        int n = nums.size();
		sort(nums.begin(), nums.end());
		  for (int i = 0; i < n - 1; i++) 
		  {
		    if (nums[i] == nums[i + 1]) 
		    {
		      return nums[i];
		    }
		  }
        return -1;
}
    
};
```

## Approach 3 : Linked List Cycle Method

![[Pasted image 20231218145249.png]]

```cpp
class Solution {
public:
    int findDuplicate(vector<int>& nums) 
    {
        ios_base::sync_with_stdio(false);
        cin.tie(NULL);
        
        int slow = nums[nums[0]];
        int fast = nums[nums[nums[0]]];
        
        while (slow != fast) 
        {
            slow = nums[slow];
            fast = nums[nums[fast]];
        }
        
        slow = nums[0];
        while (slow != fast) 
        {
            slow = nums[slow];
            fast = nums[fast];
        }
        
        return fast;
    }
};
```

# [661. Image Smoother](https://leetcode.com/problems/image-smoother/)

## Approach 1: Iterating through all the 8 adjacent cases if available

*Time Complexity* : $O(m*n)$
*Space Complexity* : $O(m*n)$

```cpp
class Solution {
public:
    vector<vector<int>> imageSmoother(vector<vector<int>>& img) 
    {
        vector<vector<int>> smooth(img.size(),vector<int>(img[0].size()));
        int m = img.size();
        int n = img[0].size();
        
        for(int i=0;i<img.size();i++)
        {
            for(int j=0;j<img[0].size();j++)
            {
                int sum=0;
                int count=0;
                for (int x = i - 1; x <= i + 1; x++) 
                {
                    for (int y = j - 1; y <= j + 1; y++) 
                    {
                        if (0 <= x && x < m && 0 <= y && y < n) 
                        {
                            sum += img[x][y];
                            count += 1;
                        }
                    }
                }

                smooth[i][j] = sum / count;
            }
        }

        return smooth;
    }

};
```

## Approach 2: Space-Optimized Smoothened Image

```cpp
class Solution {
public:
    vector<vector<int>> imageSmoother(vector<vector<int>>& img) {
        // Save the dimensions of the image.
        int m = img.size();
        int n = img[0].size();

        // Create temp array of size n.
        vector<int> temp(n);
        int prevVal = 0;

        // Iterate over the cells of the image.
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                // Initialize the sum and count 
                int sum = 0;
                int count = 0;

                // Bottom neighbors
                if (i + 1 < m) {
                    if (j - 1 >= 0) {
                        sum += img[i + 1][j - 1];
                        count += 1;
                    }
                    sum += img[i + 1][j];
                    count += 1;
                    if (j + 1 < n) {
                        sum += img[i + 1][j + 1];
                        count += 1;
                    }
                }

                // Next neighbor
                if (j + 1 < n) {
                    sum += img[i][j + 1];
                    count += 1;
                }
                
                // This cell
                sum += img[i][j];
                count += 1;

                // Previous neighbor
                if (j - 1 >= 0) {
                    sum += temp[j - 1];
                    count += 1;
                }

                // Top neighbors
                if (i - 1 >= 0) {
                    // Left-top corner-sharing neighbor.
                    if (j - 1 >=  0) {
                        sum += prevVal;
                        count += 1;
                    }
                    
                    // Top edge-sharing neighbor.
                    sum += temp[j];
                    count += 1;

                    // Right-top corner-sharing neighbor.
                    if (j + 1 < n) {
                        sum += temp[j + 1];
                        count += 1;
                    }
                }

                // Store the original value of temp[j], which represents
                // original value of img[i - 1][j].
                if (i - 1 >= 0) {
                    prevVal = temp[j];
                }

                // Save current value of img[i][j] in temp[j].
                temp[j] = img[i][j];

                // Overwrite with smoothed value.
                img[i][j] = sum / count;
            }
        }

        // Return the smooth image.
        return img;
    }
};
```










# [88. Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/) 🚀

```cpp
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) 
    {
        for(int i=m;i<m+n;i++)
        {
            nums1[i]=nums2[i-m];
        }
        sort(nums1.begin(),nums1.end());
    }
};
```

```cpp
void merge(long long arr1[], long long arr2[], int n, int m) 
{
    //Declare 2 pointers:
    int left = n - 1;
    int right = 0;
    //Swap the elements until arr1[left] is
    // smaller than arr2[right]:
    while (left >= 0 && right < m) 
    {
        if (arr1[left] > arr2[right]) 
        {
            swap(arr1[left], arr2[right]);
            left--, right++;
        }
        else 
        {
            break;
        }
    }
  
    sort(arr1, arr1 + n);
    sort(arr2, arr2 + m);

}
```

```cpp
#include <bits/stdc++.h>

using namespace std;

int main()
{
    int n, m;
    cin >> n >> m;
    int a[n], b[m], c[n + m];

    for (int i = 0; i < n; i++)
    {
        cin >> a[i];
    }
    for (int j = 0; j < m; j++)
    {
        cin >> b[j];
    }
    int i = 0, j = 0, k = 0;
    while (i < n && j < m)
    {
        if (a[i] < b[j])
        {
            c[k++] = a[i++];
        }
        else
        {
            c[k++] = b[j++];
        }
    }

    while (i < n)
        c[k++] = a[i++];
    while (j < m)
        c[k++] = b[j++];

    for (int i = 0; i < n + m; i++)
        cout << c[i] << " ";
    return 0;
}
```
## Approach 3: Using gap method

```cpp
#include <bits/stdc++.h>
using namespace std;

void swapIfGreater(long long arr1[], long long arr2[], int ind1, int ind2) {
    if (arr1[ind1] > arr2[ind2]) {
        swap(arr1[ind1], arr2[ind2]);
    }
}

void merge(long long arr1[], long long arr2[], int n, int m) {
    // len of the imaginary single array:
    int len = n + m;

    // Initial gap:
    int gap = (len / 2) + (len % 2);

    while (gap > 0) {
        // Place 2 pointers:
        int left = 0;
        int right = left + gap;
        while (right < len) {
            // case 1: left in arr1[]
            //and right in arr2[]:
            if (left < n && right >= n) {
                swapIfGreater(arr1, arr2, left, right - n);
            }
            // case 2: both pointers in arr2[]:
            else if (left >= n) {
                swapIfGreater(arr2, arr2, left - n, right - n);
            }
            // case 3: both pointers in arr1[]:
            else {
                swapIfGreater(arr1, arr1, left, right);
            }
            left++, right++;
        }
        // break if iteration gap=1 is completed:
        if (gap == 1) break;

        // Otherwise, calculate new gap:
        gap = (gap / 2) + (gap % 2);
    }
}

int main()
{
    long long arr1[] = {1, 4, 8, 10};
    long long arr2[] = {2, 3, 9};
    int n = 4, m = 3;
    merge(arr1, arr2, n, m);
    cout << "The merged arrays are: " << "\n";
    cout << "arr1[] = ";
    for (int i = 0; i < n; i++) {
        cout << arr1[i] << " ";
    }
    cout << "\narr2[] = ";
    for (int i = 0; i < m; i++) {
        cout << arr2[i] << " ";
    }
    cout << endl;
    return 0;
}

```

# [Repeat and Missing Number](https://takeuforward.org/data-structure/find-the-repeating-and-missing-numbers/) 🚀

## Approach 1: Brute Force

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> findMissingRepeatingNumbers(vector<int> a) {
    int n = a.size(); // size of the array
    int repeating = -1, missing = -1;

    //Find the repeating and missing number:
    for (int i = 1; i <= n; i++) {
        //Count the occurrences:
        int cnt = 0;
        for (int j = 0; j < n; j++) {
            if (a[j] == i) cnt++;
        }

        if (cnt == 2) repeating = i;
        else if (cnt == 0) missing = i;

        if (repeating != -1 && missing != -1)
            break;
    }
    return {repeating, missing};
}
int main()
{
    vector<int> a = {3, 1, 2, 5, 4, 6, 7, 5};
    vector<int> ans = findMissingRepeatingNumbers(a);
    cout << "The repeating and missing numbers are: {"
         << ans[0] << ", " << ans[1] << "}\n";
    return 0;
}


```

## Approach 2 : Using Hashing (Easy)

Count in a hash of size of maximum number than count it in hash and then find freq of 2 and 0 for following answer

## Approach 3: Using Math

X ->Repeating Number and Y -> Missing Number

Sum of First N numbers = S<sub>Expected</sub>
Given Sum = S<sub>given</sub>
S<sub>Expected</sub> -Y + X=  S<sub>given</sub>

**S<sub>Expected</sub> -  S<sub>given</sub>= X - Y  ---------- Eq 1

The summation of squares of the first N numbers is: S<sup>2</sup><sub>Expected</sub>

S2n = (N*(N+1)*(2N+1))/6

X+Y = (S2 - S2n) / (X-Y)

X+Y= (S<sup>2</sup><sub>Expected</sub> - S<sup>2</sup><sub>Given</sub>) /( **S<sub>Expected</sub> -  S<sub>given</sub>)


```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> findMissingRepeatingNumbers(vector<int> a) {
    long long n = a.size(); // size of the array

    // Find Sn and S2n:
    long long SN = (n * (n + 1)) / 2;
    long long S2N = (n * (n + 1) * (2 * n + 1)) / 6;

    // Calculate S and S2:
    long long S = 0, S2 = 0;
    for (int i = 0; i < n; i++) {
        S += a[i];
        S2 += (long long)a[i] * (long long)a[i];
    }

    //S-Sn = X-Y:
    long long val1 = S - SN;

    // S2-S2n = X^2-Y^2:
    long long val2 = S2 - S2N;

    //Find X+Y = (X^2-Y^2)/(X-Y):
    val2 = val2 / val1;

    //Find X and Y: X = ((X+Y)+(X-Y))/2 and Y = X-(X-Y),
    // Here, X-Y = val1 and X+Y = val2:
    long long x = (val1 + val2) / 2;
    long long y = x - val1;

    return {(int)x, (int)y};
}

int main()
{
    vector<int> a = {3, 1, 2, 5, 4, 6, 7, 5};
    vector<int> ans = findMissingRepeatingNumbers(a);
    cout << "The repeating and missing numbers are: {"
         << ans[0] << ", " << ans[1] << "}\n";
    return 0;
}

```

## Approach 4: Using XOR (TD💀)

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> findMissingRepeatingNumbers(vector<int> a) {
    int n = a.size(); // size of the array

    int xr = 0;

    //Step 1: Find XOR of all elements:
    for (int i = 0; i < n; i++) {
        xr = xr ^ a[i];
        xr = xr ^ (i + 1);
    }

    //Step 2: Find the differentiating bit number:
    int number = (xr & ~(xr - 1));

    //Step 3: Group the numbers:
    int zero = 0;
    int one = 0;
    for (int i = 0; i < n; i++) {
        //part of 1 group:
        if ((a[i] & number) != 0) {
            one = one ^ a[i];
        }
        //part of 0 group:
        else {
            zero = zero ^ a[i];
        }
    }

    for (int i = 1; i <= n; i++) {
        //part of 1 group:
        if ((i & number) != 0) {
            one = one ^ i;
        }
        //part of 0 group:
        else {
            zero = zero ^ i;
        }
    }

    // Last step: Identify the numbers:
    int cnt = 0;
    for (int i = 0; i < n; i++) {
        if (a[i] == zero) cnt++;
    }

    if (cnt == 2) return {zero, one};
    return {one, zero};
}

int main()
{
    vector<int> a = {3, 1, 2, 5, 4, 6, 7, 5};
    vector<int> ans = findMissingRepeatingNumbers(a);
    cout << "The repeating and missing numbers are: {"
         << ans[0] << ", " << ans[1] << "}\n";
    return 0;
}

```



# [Merge Sort](https://takeuforward.org/data-structure/merge-sort-algorithm/) 🚀

## Pre-requisite
```cpp
#include <bits/stdc++.h>

using namespace std;

int main()
{
    int n, m;
    cin >> n >> m;
    int a[n], b[m], c[n + m];

    for (int i = 0; i < n; i++)
    {
        cin >> a[i];
    }
    for (int j = 0; j < m; j++)
    {
        cin >> b[j];
    }
    int i = 0, j = 0, k = 0;
    while (i < n && j < m)
    {
        if (a[i] < b[j])
        {
            c[k++] = a[i++];
        }
        else
        {
            c[k++] = b[j++];
        }
    }

    while (i < n)
        c[k++] = a[i++];
    while (j < m)
        c[k++] = b[j++];

    for (int i = 0; i < n + m; i++)
        cout << c[i] << " ";
    return 0;
}
```

## The Merge Sort Code

```cpp
#include <bits/stdc++.h>
using namespace std;

void merge(vector<int> &arr, int low, int mid, int high)
{
    vector<int> temp; // temporary array
    int left = low;      // starting index of left half of arr
    int right = mid + 1;   // starting index of right half of arr

    //storing elements in the temporary array in a sorted manner//
    cout<<"Before Sorting: "<<endl;
    cout << "Left half: ";
    for (int i = 0; i < mid - low + 1; i++)
        cout << arr[i] << " ";

  
    cout << "Right half: ";
    for (int i = mid - low + 1; i < high - low + 1; i++)
        cout << arr[i] << " ";
  
    cout << endl;

  

    while (left <= mid && right <= high)
    {
        if (arr[left] <= arr[right])
        {
            temp.push_back(arr[left]);
            left++;
        }
        else
        {
            temp.push_back(arr[right]);
            right++;
        }
    }
  
    // if elements on the left half are still left //
    while (left <= mid)
    {
        temp.push_back(arr[left]);
        left++;
    }
    //  if elements on the right half are still left //
    while (right <= high)
    {
        temp.push_back(arr[right]);
        right++;
    }
  
    cout<<"After Sorting: "<<endl;
    cout << "Left half: ";
    
    for (int i = 0; i < mid - low + 1; i++)
        cout << temp[i] << " ";

  
    cout << "Right half: ";
    for (int i = mid - low + 1; i < high - low + 1; i++)
        cout << temp[i] << " ";

  

    cout << "Merged array: ";
    for (int i = 0; i < high - low + 1; i++)
        cout << temp[i] << " ";

  
    cout << endl;

  

    // transferring all elements from temporary to arr //
    for (int i = low; i <= high; i++)
    {
        arr[i] = temp[i - low];
    }
}

  
void mergeSort(vector<int> &arr, int low, int high)
{
    if (low >= high)
    {
        return;
    }

    int mid = (low + high) / 2 ;

    mergeSort(arr, low, mid);  // left half

    mergeSort(arr, mid + 1, high); // right half

    merge(arr, low, mid, high);  // merging sorted halves
}

  

int main() {

    vector<int> arr = {9, 4, 7, 6, 3, 1, 5}  ;
    int n = 7;
  
    cout << "Before Sorting Array: " << endl;
    for (int i = 0; i < n; i++)
    {
        cout << arr[i] << " "  ;
    }
    cout << endl;
    mergeSort(arr, 0, n - 1);
    cout << "After Sorting Array: " << endl;
    for (int i = 0; i < n; i++)
    {
        cout << arr[i] << " "  ;
    }
    cout << endl;
    return 0 ;
}
```

## Re-written

![[Pasted image 20231219172149.png]]


# [Inversion of Array (Pre-req: Merge Sort)](https://takeuforward.org/data-structure/count-inversions-in-an-array/)

## Approach 1: Brute Force

*Time Complexity* : $O(n$ <sup>2</sup>$)$
*Space Complexity* : $O(n)$

```cpp
#include <bits/stdc++.h> 

long long getInversions(long long *arr, int n)

{
    // Write your code here.
    long long a=0;

    for(int i=0;i<n;i++)
    {
        for(int j=i+1;j<n;j++)
        {
            if(arr[i]>arr[j])
            {
                a++;
            }
        }
    }

  

    return a;

}
```

## Approach 2: Using Merge Sort

*Time Complexity* : $O(n)$
*Space Complexity* : $O(n)$

```cpp

#include <bits/stdc++.h>
using namespace std;

int merge(vector<int> &arr, int low, int mid, int high) {
    vector<int> temp; // temporary array
    int left = low;      // starting index of left half of arr
    int right = mid + 1;   // starting index of right half of arr

    //Modification 1: cnt variable to count the pairs:
    int cnt = 0;

    //storing elements in the temporary array in a sorted manner//

    while (left <= mid && right <= high) {
        if (arr[left] <= arr[right]) {
            temp.push_back(arr[left]);
            left++;
        }
        else {
            temp.push_back(arr[right]);
            cnt += (mid - left + 1); //Modification 2
            right++;
        }
    }

    // if elements on the left half are still left //

    while (left <= mid) {
        temp.push_back(arr[left]);
        left++;
    }

    //  if elements on the right half are still left //
    while (right <= high) {
        temp.push_back(arr[right]);
        right++;
    }

    // transfering all elements from temporary to arr //
    for (int i = low; i <= high; i++) {
        arr[i] = temp[i - low];
    }

    return cnt; // Modification 3
}

int mergeSort(vector<int> &arr, int low, int high) {
    int cnt = 0;
    if (low >= high) return cnt;
    int mid = (low + high) / 2 ;
    cnt += mergeSort(arr, low, mid);  // left half
    cnt += mergeSort(arr, mid + 1, high); // right half
    cnt += merge(arr, low, mid, high);  // merging sorted halves
    return cnt;
}

int numberOfInversions(vector<int>&a, int n) {

    // Count the number of pairs:
    return mergeSort(a, 0, n - 1);
}

int main()
{
    vector<int> a = {5, 4, 3, 2, 1};
    int n = 5;
    int cnt = numberOfInversions(a, n);
    cout << "The number of inversions are: "
         << cnt << endl;
    return 0;
}

```
















# [74. Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)

## Approach 1: Brute Force

*Time Complexity* : $O(N*M)$
*Space Complexity* : $O(1)$

```cpp
class Solution {

public:

    bool searchMatrix(vector<vector<int>>& matrix, int target)
    {
        int n = matrix.size(), m = matrix[0].size();
        //Brute Force

        for (int i = 0; i < n; i++)

        {
            for (int j = 0; j < m; j++)
            {
                if (matrix[i][j] == target)
                    return true;
            }
        }
        return false;
    }

};
```

## Approach 2: Binary Search on each row 

Assuming that the 


*Time Complexity* : $O(N * log M)$
*Space Complexity* : $O(1)$

```cpp
class Solution 
{
bool checkFind(vector<int> arr,int target,int highy)
{
    int low=0;
    int high=highy-1;
    int mid;
    while(low<=high)
    {
        mid=(low+high)/2;
        if(arr[mid]==target)
        {
            return true;
        }
        else if(arr[mid]<target)
        {
            low=mid+1;
        }
        else
        {
            high=mid-1;
        }
    }
    return false;
}
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) 
    {
        int n = matrix.size();
        int m = matrix[0].size();
        
        //Assume Sorted
        for(vector<int> i: matrix)
        {
            if(i[0]<=target && i[m-1]>=target)
            {
                return checkFind(i,target,m); 
            }
        }

        return false;

    }
};
```

## Approach 3: Making an 2D array as 1D with help of simulation

*Time Complexity* : $O(log(m*n))$
*Space Complexity* : $O(1)$

Main Approach:
`int mid = left + (right - left) / 2;`
`matrix[mid / n] [mid % n];`
 ![[2D1-768x248.webp]]

```cpp
class Solution 
{
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) 
    {
        int n = matrix.size();
        int m = matrix[0].size();

        //Apply binary search
        int low = 0, high = n * m - 1;

        while (low <= high) 
        {
            int mid = (low + high) / 2;
            int row = mid / m, col = mid % m;
            if (matrix[row][col] == target)
            {
	            return true;
            } 
            else if (matrix[row][col] < target) 
            { 
            low = mid + 1;
            }
            else 
            {
	             high = mid - 1;
            }
        }
        return false;
    }
    
};
```









# [169. Majority Element n/2 🚀](https://leetcode.com/problems/majority-element/)

## Approach 1: Using Hash Map

*Time Complexity* : $O(n)$
*Space Complexity* : $O(n)$

```cpp
class Solution {
public:
    int majorityElement(vector<int>& nums) 
    {
        unordered_map<int,int> freq;
        int n=nums.size();

        for(int i:nums)
        {
            freq[i]++;
            if(freq[i]>n/2)
            {
                return i;
            }
        }

        return 1;
    }
};
```

## Approach 2: Moore's Voting Algo 


```cpp
class Solution {
public:
    int majorityElement(vector<int>& nums) 
    {
        int n=nums.size();
        int c=0;
        int ans=0;

        for(int i:nums)
        {
            if(c==0)
            {
                ans=i;
            }
            
            if(i!=ans)
            {
                c--;
            }
            else
            {
                c++;
            }
        
        }

        return ans;
    }
};
```






# [229. Majority Element II](https://leetcode.com/problems/majority-element-ii/) 🚀

## Approach 1: Using Hash Map as previous sum

only change is n/2 -> n/3, that's all

## Approach 2: Boyer-Moore Voting Algo



# [1071. Greatest Common Divisor of Strings](https://leetcode.com/problems/greatest-common-divisor-of-strings/)
