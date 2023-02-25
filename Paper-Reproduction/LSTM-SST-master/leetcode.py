from typing import List


class Solution:
    def backspaceCompare(self, S: str, T: str) -> bool:
        S = list(S)
        T = list(T)
        i = 0
        while i < len(S):
            if S[i] == '#':
                S.pop(i)
                if i != 0:
                    S.pop(i - 1)
                    i = i - 1
                continue
            i += 1
        j = 0
        while j < len(T):
            if T[j] == '#':
                T.pop(j)
                if j != 0:
                    T.pop(j - 1)
                    j = j - 1

                continue

            j += 1

        for a, b in zip(S, T):
            if a != b:
                return False
        return True

res=Solution()
d=res.backspaceCompare("isfcow#",
"isfco#w#")
print(d)