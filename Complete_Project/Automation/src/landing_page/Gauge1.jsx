import React, { useState, useEffect } from 'react';
import GaugeComponent from 'react-gauge-component';

function Gauge1() {
  const [humidity, setHumidity] = useState(0); // Use humidity instead of moisture
  const [humidityStatus, setHumidityStatus] = useState('');

  // Fetch the humidity data from ESP32
  useEffect(() => {
    const fetchHumidityData = async () => {
      try {
        const response = await fetch('http://192.168.43.245/readSensor'); // Replace with your ESP32 IP
        if (response.ok) {
          const data = await response.json();
          
          // Ensure data contains valid humidity and status
          if (data && typeof data.humidity === 'number') {
            setHumidity(data.humidity);
            setHumidityStatus(data.humidityStatus || 'Unknown Status');
          } else {
            console.warn("Invalid humidity data received");
            setHumidity(0); // Default fallback
          }
        } else {
          console.error("Error: Received non-200 status code.");
        }
      } catch (error) {
        console.error("Error fetching humidity data:", error);
      }
    };

    const intervalId = setInterval(fetchHumidityData, 1800); // Poll every 1.8 seconds
    return () => clearInterval(intervalId); // Clean up on component unmount
  }, []);

  return (
    <div className="gauge1-container">
      {/* Gauge to display humidity value */}
      <GaugeComponent
        value={(humidity || 0).toString()} // Display humidity value safely
        type="radial"
        labels={{
          tickLabels: {
            type: "inner",
            ticks: [
              { value: 0 },
              { value: 20 },
              { value: 40 },
              { value: 60 },
              { value: 80 },
              { value: 100 }
            ]
          }
        }}
        arc={{
          colorArray: ['#5BE12C', '#EA4228'], // Colors for humidity levels
          // subArcs: [{ limit: 30 }, { limit: 60 }, { limit: 100 }], // Adjust limits based on humidity
          subArcs: [{ limit: 20 }, { limit: 40 }, {}, {}, {}],
          padding: 0.02,
          width: 0.3
        }}
        pointer={{
          elastic: true,
          animationDelay: 0
        }}
      />

      {/* Display humidity status */}
      <br />
      <h4 className="white-h4">Humidity Status: {humidityStatus}</h4>
    </div>
  );
}

export default Gauge1;
