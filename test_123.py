from flask import Flask,render_template

app = Flask(__name__)


@app.route("/")
def home():
    return "Hello, World!"

@app.route("/login")
def home_1():
    return render_template('login.html')


if __name__ == "__main__":
    app.run(debug=True)