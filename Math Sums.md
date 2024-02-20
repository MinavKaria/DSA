
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








