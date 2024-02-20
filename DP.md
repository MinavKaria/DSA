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


