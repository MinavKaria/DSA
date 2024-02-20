
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
```