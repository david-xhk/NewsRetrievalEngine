import flask
from flask.helpers import send_from_directory
from flask import render_template, request
import pandas as pd

app = flask.Flask(__name__)
app.config["DEBUG"] = True

ARTICLES = ["Lucas","John"]

df = pd.read_csv('fake_data.csv')
SELECTED_ARTICLES = []
for idx, row in df.iterrows():
  content = row["content"][:300] + "..."
  date = row["publish_date"]
  print(type(date))
  if type(date) == str:
    date = date.split()[0]
  else:
    date = "Not available"
  SELECTED_ARTICLES.append({"id": row["ID"], "source": row["source"], "title": row["title"], "content": content, "date": date})


@app.route('/', methods=['GET', 'POST'])
def home():
  if request.method == "POST":
    ARTICLES.append("helkj")
  return render_template("index.html", articles=SELECTED_ARTICLES)

app.run()