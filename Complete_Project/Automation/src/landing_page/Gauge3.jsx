import React, { useState, useEffect } from 'react';
import GaugeComponent from 'react-gauge-component';

function Gauge3() {
  const [co2Level, setCO2Level] = useState(0); // CO2 level value
  const [co2Status, setCO2Status] = useState('');

  // Fetch the CO2 level data from ESP32
  useEffect(() => {
    const fetchCO2Data = async () => {
      try {
        const response = await fetch('http://192.168.43.245/readSensor'); // Replace with your ESP32 IP
        if (response.ok) {
          const data = await response.json();

          // Ensure data contains valid CO2 level and status
          if (data && typeof data.co2Level === 'number') {
            setCO2Level(data.co2Level);
            setCO2Status(data.co2Status || 'Unknown Status');
          } else {
            console.warn("Invalid CO2 level data received");
            setCO2Level(0); // Default fallback
          }
        } else {
          console.error("Error: Received non-200 status code.");
        }
      } catch (error) {
        console.error("Error fetching CO2 level data:", error);
      }
    };

    const intervalId = setInterval(fetchCO2Data, 1800); // Poll every 1.8 seconds
    return () => clearInterval(intervalId); // Clean up on component unmount
  }, []);

  return (
    <div className="gauge-co2-container">
      {/* Gauge to display CO2 level value */}
      <GaugeComponent
        value={(co2Level || 0).toString()} // Display CO2 level value safely
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
              { value: 100 },
            ]
          }
        }}
        arc={{
          colorArray: ['#5BE12C', '#FFA500', '#EA4228'], // Colors for CO2 levels
          subArcs: [{ limit: 20 }, { limit: 40 }, {}, {}, {}], // Adjust limits based on CO2 levels
          padding: 0.02,
          width: 0.3
        }}
        pointer={{
          elastic: true,
          animationDelay: 0
        }}
      />

      {/* Display CO2 status */}
      <br />
      <h4 className="white-h4">Air Qualtiy Status: {co2Status}</h4>
    </div>
  );
}

export default Gauge3;
