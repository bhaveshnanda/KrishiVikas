from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/index.html', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/contactUs.html', methods=['GET', 'POST'])
def contact():
    return render_template('contactUs.html')


@app.route('/cultivation.html', methods=['GET', 'POST'])
def cultivation():
    return render_template('cultivation.html')


@app.route('/message.html', methods=['GET', 'POST'])
def message():
    return render_template('message.html')


@app.route('/messag2.html', methods=['GET', 'POST'])
def messag2():
    return render_template('messag2.html')


@app.route('/news.html', methods=['GET', 'POST'])
def news():
    return render_template('news.html')


@app.route('/rent_machine.html', methods=['GET', 'POST'])
def rent_machine():
    return render_template('rent_machine.html')


if __name__ == '__main__':
    app.run(debug=True, port=7000)
