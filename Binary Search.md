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
