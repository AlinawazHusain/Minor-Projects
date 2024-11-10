from AST import master , print_rule , AstNode
import warnings
warnings.filterwarnings('ignore')
def parser(root :AstNode , person:dict)->bool:
    if(root == None):
        return True
    
    if(root.type == "OPERAND"):
        data = root.value.split()
        attribute = data[0]
        op = data[1]
        val1 = data[2]
        for i in person.keys():
            if(str(i).upper() == attribute):
                val2 = str(person[i]).upper()
                if(op == "="):
                    return val1 == val2
                elif(op == '!='):
                    return not val1==val2
                elif(op == '>'):
                    return int(val2)>int(val1)
                elif(op == '<'):
                    return int(val2)<int(val1)
                elif(op == '<='):
                    return int(val2)<=int(val1)
                elif(op == '>='):
                    return int(val2)>=int(val1)
        return False

    else:
        left = False
        right = False
        if(root.left != None):
            left = parser(root.left , person)
        if(root.right != None):
            right = parser(root.right , person)
        
        if(root.value == "AND"):
            return left and right
        else:
            return left or right
        

if __name__ == "__main__":
    s = "(department != sales)"
    root = master(s)
    rules = print_rule(root)
    print(rules)
    d = {"age": 40,"department": "sales", "salary": 400, "experience": 3}
    res = parser(root , d)
    print(res)
