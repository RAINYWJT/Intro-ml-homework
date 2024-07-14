import math
def cal(a,b): 
              return -(a*math.log2(a) + b *math.log2(b))

print(cal(1/4,3/4))