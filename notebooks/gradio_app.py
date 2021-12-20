import gradio as gr
import pandas as pd
import joblib

import configparser
config = configparser.ConfigParser()
config.read("config.ini")


def return_output(text):

  output = predict(text)
  output_string = f"Device: {output[0]}\n\nProbability: {output[1]}"
  return output_string


def predict(text):

  model = joblib.load(config['MODELS']['nb_model'])
  data = pd.DataFrame({"text":[text]})
  #prediction_class = model.predict(data)[0]
  #prediction_prob = round(model.predict_proba(data).max(), 3)
  #return prediction_class, prediction_prob
  pred = model.predict_proba(data)[0]
  return {'Android': pred[0], 'iPhone': pred[1]}


description = "According to the dataset used for this model, Trump  mainly uses two devices for tweeting - an Android and an iPhone device.\nIt seems likely that members of his staff are tweeting on his behalf using iPhone.\nTry and see if you can write an 'iPhone' and an 'Android' tweet."

iface = gr.Interface(fn=predict,
                     #fn=return_output,
                     inputs="text",
                     #outputs="text",
                     outputs="label",
                     allow_flagging=False, title="iPhone or Android?",
                     interpretation="default", description=description)

iface.launch()