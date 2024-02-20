Syntax:
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
