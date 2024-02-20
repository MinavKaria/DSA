

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

