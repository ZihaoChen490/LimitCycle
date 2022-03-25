class Solution:
    def Node(self,val=None):
        self.val=None
    def isBalance(self,root):
        def Depth(root):
            if not root:
                return 0
            return max(Depth(root.left),Depth(root.right))
        if not root:
            return 1
        leftDeep=Depth(root.left)
        rightDeep=Depth(root.right)
        if abs(leftDeep-rightDeep)>1:
            return 0
        return self.isBalance(root.right) and self.isBalance(root.left)

