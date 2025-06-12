from flask import Flask, render_template_string
import serial

app = Flask(__name__)

# Set up the serial connection
ser = serial.Serial('/dev/tty.usbmodem14701', 9600)# Replace with your Arduino's serial port

@app.route('/')
def index():
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').rstrip()
    else:
        line = "Waiting for data..."
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        
        <head>
            <title>Bluetooth Data</title>
        </head>
        <body>
            <h1>Received Data:</h1>
            <p>{{ data }}</p>
        </body>
        </html>
        ''', data=line)
if __name__ == '__main__':
    app.run(debug=True)
