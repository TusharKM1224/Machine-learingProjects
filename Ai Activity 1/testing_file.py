import joblib
from flask import Flask ,jsonify,request
 
feedback_model=joblib.load("Feedback_model")

app=Flask(__name__)
@app.route('/')
def index():
    return "hello world "
@app.route('/feedback_mod',methods =['POST'])
def feedback_mod():
    msg=request.form.get('msg')
    response=feedback_model.predict([msg])
    result={"response":str(response)}
    return jsonify(result) 
if __name__=='__main__':
    app.run(debug=True)




 