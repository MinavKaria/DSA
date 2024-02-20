 ```cpp
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




