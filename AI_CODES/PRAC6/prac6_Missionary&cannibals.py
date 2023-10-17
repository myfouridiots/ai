um = 3  # unreached Missionaries
uc = 3  # unreached Cannibals
rm = 0  # reached Missionaries
rc = 0  # reached Cannibals
flag = 0  # for checking where currently the boat is present (on start or at end)
select = 0

# for displaying the current state of the problem
def display(p1, p2):
    for i in range(rm):
        print('M', end=' ')
    for i in range(rc):
        print('C', end=' ')
    print('|', end='')
    if flag == 0:
        print("------water------ %c , %c/--" % (p1, p2), end=' ')
    else:
        print("--%c , %c/------water------" % (p1, p2), end=' ')
    print("|", end='')
    for i in range(um):
        print('M', end=' ')
    for i in range(uc):
        print('C', end=' ')
    print('')

# for checking if all Missionaries and Cannibals are reached
def isreached():
    if rm == 3 and rc == 3:
        return 0
    return 1

# the main algorithm working
def solve():
    global um
    global uc
    global rm
    global rc
    global flag
    global select
    while isreached():
        if flag == 0:
            if select == 1:
                display('C', ' ')
                uc += 1
            if select == 2:
                display('C', 'M')
                um += 1
                uc += 1
            if ((um - 2) >= uc and (rm + 2) >= rc) or (um - 2) == 0:
                um -= 2
                select = 1
                display('M', 'M')
                flag = 1
            elif ((uc - 2) < um and (rm == 0 or (rc + 2) <= rm)) or um == 0:
                uc = uc - 2
                select = 2
                display('C', 'C')
                flag = 1
            elif (uc - 1 <= um - 1) and (rm + 1 >= rc + 1):
                um -= 1
                uc -= 1
                select = 3
                display('M', 'C')
                flag = 1
        else:
            if select == 1:
                display('M', 'M')
                rm += 2
            if select == 2:
                display('C', 'C')
                rc += 2
            if select == 3:
                display('M', 'C')
                rc += 1
                rm += 1
            if isreached():
                if (rc > 1 and rm == 0) or um == 0:
                    rc -= 1
                    select = 1
                    display('C', ' ')
                    flag = 0
                elif (uc + 2) > um:
                    rc -= 1
                    rm -= 1
                    select = 2
                    display('C', 'M')
                    flag = 0

def main():
    print("Missionaries And Cannibal Problem")
    display(' ', ' ')
    solve()
    display(' ', ' ')

if __name__ == "__main__":
    main()
