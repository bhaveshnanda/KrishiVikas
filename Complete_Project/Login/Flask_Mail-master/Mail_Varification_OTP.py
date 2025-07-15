# Email Varification Using OTP in Flask

from flask import Flask, render_template, request, jsonify, redirect
from flask_mail import Mail, Message
from random import randint
import pandas as pd
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

mail = Mail(app)

# Configure Flask app
app.config["MAIL_SERVER"] = 'smtp.gmail.com'
app.config["MAIL_PORT"] = 465
app.config["MAIL_USERNAME"] = 'nandabhavesh2003@gmail.com'
app.config['MAIL_PASSWORD'] = "xgfe ekwz nymu xwrv"  # you have to give your password of gmail account
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

# Initialize Flask-Mail
mail = Mail(app)

# Generate OTP
otp = randint(000000, 999999)

# Get the current directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# Specify the file name and path for Excel file
excel_file_path = os.path.join(current_dir, 'data.xlsx')

# Read existing data from the Excel file
try:
    userdetails = pd.read_excel(excel_file_path, engine='openpyxl')
except FileNotFoundError:
    userdetails = pd.DataFrame(columns=['Email', 'Name', 'City', 'State'])  # Create empty DataFrame if file not found

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/verify', methods=["POST"])
def verify():
    email = request.form['email']
    if email in userdetails['Email'].values:
        # Compose the email body with an additional message
        msg = Message(subject='OTP', sender='pandeyritik527@gmail.com', recipients=[email])
        msg.body = f"Your login OTP is: {otp}\n\nThis is your login OTP. Don't share it with anyone."
        mail.send(msg)
        return render_template('otp.html')
    else:
        error_msg = "Invalid user"
        return render_template('index.html', error=error_msg)


@app.route('/validate', methods=['POST'])
def validate():
    user_otp = request.form['otp']
    if otp == int(user_otp):
        return redirect('http://localhost:7000/')
    else:
        error_msg = "Incorrect OTP"
        return render_template('otp.html', error=error_msg)

@app.route('/data', methods=['GET', 'POST'])
def process_data():
    global userdetails
    if request.method == 'POST':
        data_from_html = request.form
        email = data_from_html.get('email_input')
        name = data_from_html.get('name_input')
        city = data_from_html.get('city_input')
        state = data_from_html.get('state_input')
        
        login_data = {
            'Email': [email],
            'Name': [name],
            'City': [city],
            'State': [state]
        }
        userdetails = pd.concat([userdetails, pd.DataFrame(login_data)], ignore_index=True)
        try:
            userdetails.to_excel(excel_file_path, index=False, engine='openpyxl')
            return jsonify(success=True, error="Account Created Successfully🎉 You can now Login")
        except Exception as e:
            return jsonify(success=False, error=str(e))
    return redirect('http://localhost:7000/')

@app.route('/register.html', methods=['GET', 'POST'])
def register():
    return render_template('register.html')

@app.route('/index.html', methods=['GET', 'POST'])
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
