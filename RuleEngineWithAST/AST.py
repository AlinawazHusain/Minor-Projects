class AstNode:
    def __init__(self , type , value):
        self.type = type
        self.value = value
        self.left = None
        self.right = None
    
    def __repr__(self):
        return f"{self.value}"


def create_rule(s:str)->AstNode:
    s.strip()
    if not s:
        return None
    if(s.find("AND") != -1):
        root = AstNode("OPERATOR" , "AND")
        idx = s.find("AND")
        root.left = create_rule(s[:idx].strip())
        root.right = create_rule(s[idx+3:].strip())
        return root
    
    elif(s.find("OR") != -1):
        root = AstNode("OPERATOR" , "OR")
        idx = s.find("OR")
        root.left = create_rule(s[:idx].strip())
        root.right = create_rule(s[idx+2:].strip())
        return root 
    else :
        input_val = ""
        for i in s:
            if(i == "'"):
                continue
            input_val+= i
        root = AstNode("OPERAND" , input_val.upper())
        return root

def master(s:str)->AstNode:
    if(s.find('(') == -1):
        return create_rule(s)
    else:
        l = split_in_two(s)
        if len(l) == 1:
            return create_rule(l[0])
        
        else:
            root = AstNode("OPERATOR" , l[1])
            root.left = master(l[0])
            root.right = master(l[2])
            return root



def print_rule(root: AstNode)->str:
    if(root == None):
        return ""
    s = ""
    s+= print_rule(root.left)
    s+= repr(root)+" "
    s+= print_rule(root.right)
    return s


def split_in_two(s:str)->list:
    s = s.upper()
    l = []
    open = 0
    firstIndex = -1
    for c in range(len(s)):
        if(s[c] == '('):
            if(firstIndex == -1):
                firstIndex = c
            open+=1
        elif(s[c] == ')'):
            if open>0:
                open-=1
                if(open == 0):
                    s1 = s[firstIndex+1:c]
                    l.append(s1)
                    s2_temp = s[c+1:].strip()
                    if len(s2_temp) !=0:
                        op = s2_temp[:3] if s2_temp[:3] == "AND" else "OR"
                        s2 = s2_temp[3:].strip() if op == "AND" else s2_temp[2:]
                        l.append(op)
                        l.append(s2)
                    return l
            else :
                print("invalid string")
                return l
    return l

if __name__ == "__main__":
    rule = "age >30 AND department = finance"
    # rule1 = "((age > 30 AND department = 'Sales') OR (age < 25 AND department = 'Marketing')) AND (salary > 50000 OR experience >5)"
    root = master(rule)
    s = print_rule(root)
    print(s)
    # l = split_in_two(rule1)
    # ll = split_in_two(l[0])
    # lr = split_in_two(ll[2])
    # print(l)
    # print(ll)
    # print(lr)
