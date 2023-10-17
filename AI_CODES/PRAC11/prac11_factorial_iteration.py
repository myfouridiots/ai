def fact(n):
    if n<1:
        return 1
    else:
        f=n*fact(n-1)
        return f
num=int(input("enter:"))
print(fact(num))
fact(num)
