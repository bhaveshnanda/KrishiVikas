import subprocess
import threading
import re

# Function to execute a command
def execute_command(command):
    try:
        # Execute the command and capture the output
        result = subprocess.run(command, shell=True, check=True,capture_output=True, text=True)
        
        # Extract and print port information if available
        port_match = re.search(r'port\s(\d+)', result.stdout)
        if port_match:
            print(f"Command '{command}' is running on port {port_match.group(1)}")
        else:
            print(f"Command '{command}' output:\n{result.stdout}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

# Define the commands
commands = [
    # "cd Agriculture-Machinery-Rental-Website-main/Login/Flask_Mail-master && python Mail_Varification_OTP.py",
    # "cd Agriculture-Machinery-Rental-Website-main/shopping && yarn start",
    "cd Agriculture-Machinery-Rental-Website-main/Crop-and-Fertilizer-Recommandation-System-using-FLASK-main && python app.py",
    "cd Agriculture-Machinery-Rental-Website-main/Crop_Prediction && python app.py",
    "cd Agriculture-Machinery-Rental-Website-main/AirBnb && npm run dev",
    "cd Agriculture-Machinery-Rental-Website-main/Crop_website && npm start",
    "cd Agriculture-Machinery-Rental-Website-main/main && python backend.py",
    "cd Agriculture-Machinery-Rental-Website-main/Login/Flask_Mail-master && python Mail_Varification_OTP.py",
    "cd Agriculture-Machinery-Rental-Website-main/Automation && npm run dev"
    
]

# Create threads to run each command
threads = []
for command in commands:
    thread = threading.Thread(target=execute_command, args=(command,))
    threads.append(thread)

# Start the threads
for thread in threads:
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

print("All commands have finished executing.")


