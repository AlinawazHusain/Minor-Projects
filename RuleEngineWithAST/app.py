from flask import Flask, render_template, request
import main
import AST
import saveInDB
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    rule = request.form['rule']
    data = request.form['data']
    
    # Convert string representation of dictionary to actual dictionary
    try:
        data_dict = eval(data)  # Use eval carefully; better to use json.loads for safety
    except Exception as e:
        return f"Error in data format: {e}"

    # Here you can process the rule and data_dict as needed
    root = AST.master(rule)
    res = main.parser(root , data_dict)
    rule = AST.print_rule(root)
    try:
        saveInDB.init_db()
        saveInDB.save_rule(rule)
    except:
        print("error in saving rule")

    if res == True:
        return f"valid"
    else:
        return f"invalid"

if __name__ == '__main__':
    app.run(debug=True)
