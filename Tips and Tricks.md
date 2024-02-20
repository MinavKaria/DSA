A spell to make the code run faster
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
```