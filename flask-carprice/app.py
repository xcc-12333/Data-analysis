from flask import Flask, render_template, request
import pandas as pd
import model
app = Flask("cantine_predict")

def get_predict_result(df):
    # datajson = """{"pizza":{"4047":180},"egg-tomato":{"4047":435},"bean":{"4047":286},"Fish in oil":{"4047":78},"coca_cola":{"4047":546},"humberger":{"4047":694},"franch_frise":{"4047":685},"dessert":{"4047":477},"family_portrait_package":{"4047":40},"dumplings":{"4047":6},"business volume":{"4047":28990}}"""
    # df = pd.read_json(datajson)
    num_columns = ['pizza', 'egg-tomato', 'bean', 'Fish in oil', 'coca_cola', 'humberger',
                   'franch_frise', 'dessert', 'family_portrait_package', 'dumplings']
    load_num_fatures = model.carPriceModel.scaler.transform(df[num_columns])

    load_final_fatures = load_num_fatures
    result = model.carPriceModel.predictor.predict(load_final_fatures)



    print(result)
    return result

@app.route("/predict", methods=["get", "post"])
def predict():
    result = None
    if request.method == "POST":
        data = dict(request.form)
        df = pd.DataFrame([data.values()], columns=data.keys())
        #return "餐厅营业额预测值：" + str(get_predict_result())
        result = str(get_predict_result(df))
    return render_template("predict.html", result=result)

app.run(host="0.0.0.0", port=5020)