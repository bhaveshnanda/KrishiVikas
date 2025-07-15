import React, { useState, useEffect } from 'react';
import Thermometer from 'react-thermometer-component';
import '../index.css';

function Thermometergauge() {
  const [temperature, setTemperature] = useState(0); // State to store temperature

  // Fetch the temperature data from ESP32
  useEffect(() => {
    const fetchTemperatureData = async () => {
      try {
        const response = await fetch('http://192.168.43.245/readSensor'); // Replace with your ESP32 IP
        if (response.ok) {
          const data = await response.json();
          setTemperature(data.temperature); // Set temperature value from fetched data
        } else {
          console.error("Error: Received non-200 status code.");
        }
      } catch (error) {
        console.error("Error fetching temperature data:", error);
      }
    };

    const intervalId = setInterval(fetchTemperatureData, 1800); // Poll every 1.8 seconds
    return () => clearInterval(intervalId); // Clean up on component unmount
  }, []);

  return (
    <center>
      <Thermometer
        theme="dark"
        value={temperature} // Use dynamic temperature value here
        max="80"
        steps="3"
        format="°C"
        size="large"
        height="350"
      />
    </center>
  );
}

export default Thermometergauge;
