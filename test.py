'''
    作者:LSY
    文件:test
    日期:2022/4/13 17:56
    版本:
    功能:
    需求分析:
5 1
L1-1 is a qianddao problem.
L1-2 is so...eadsy.
L1-3 is Eadsy.
L1-4 is qianDao.
Wow, such L1-5, so easy.

'''
import re
n,m=map(int , input().split())
pattern1 = r"easy"
pattern2 = r"qiandao"
b = list()
for i in range(n):
    a = input()
    b.append(a)
j = 0
for i in range(n):
    if re.search(pattern1, b[i]) != None or re.search(pattern2, b[i]) != None:
        j+=1
    else:
        if(m>0):
            j+=1
            m-=1
print(j)
if(j<n):
    print(b[j-1])
else:
    print("Wo AK le")