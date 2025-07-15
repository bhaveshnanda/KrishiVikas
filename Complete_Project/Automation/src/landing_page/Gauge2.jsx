import React, { useState, useEffect } from 'react';
import GaugeComponent from 'react-gauge-component';

function Gauge2() {
  const [moisture, setMoisture] = useState(0);
  const [soilStatus, setSoilStatus] = useState('');
  const [pumpState, setPumpState] = useState(false); // Assuming pumpState is a boolean
  const [isToggling, setIsToggling] = useState(false); // State to manage toggle button status

  // Fetch the soil moisture data from ESP32
  useEffect(() => {
    const fetchMoistureData = async () => {
      try {
        const response = await fetch('http://192.168.43.245/readSensor'); // Replace with your ESP32 IP
        if (response.ok) {
          const data = await response.json();
          if (data) {
            setMoisture(data.soilMoisture || 0); // Ensure fallback to avoid errors
            setSoilStatus(data.soilStatus || 'Unknown');
            setPumpState(data.pumpState || false);
          } else {
            console.warn('No data received from ESP32.');
          }
        } else {
          console.error('Error: Received non-200 status code.');
        }
      } catch (error) {
        console.error('Error fetching moisture data:', error);
      }
    };

    const intervalId = setInterval(fetchMoistureData, 1800); // Poll every 1.8 seconds
    return () => clearInterval(intervalId); // Clean up on component unmount
  }, []);

  // Handle the pump toggle request
  const togglePump = async () => {
    setIsToggling(true); // Indicate that the pump toggle is in progress
    try {
      const response = await fetch('http://192.168.43.245/togglePump'); // Replace with your ESP32 IP
      if (response.ok) {
        setPumpState((prevState) => !prevState); // Toggle the pump state upon success
      } else {
        console.error('Error toggling pump: Received non-200 status code.');
      }
    } catch (error) {
      console.error('Error toggling pump:', error);
    } finally {
      setIsToggling(false); // Reset the toggling state
    }
  };

  return (
    <div className="gauge1-container">
      {/* Gauge to display soil moisture value */}
      <GaugeComponent
        value={moisture.toString()}
        type="radial"
        labels={{
          tickLabels: {
            type: 'inner',
            ticks: [
              { value: 0 },
              { value: 20 },
              { value: 40 },
              { value: 60 },
              { value: 80 },
              { value: 100 },
            ],
          },
        }}
        arc={{
          colorArray: ['#5BE12C', '#EA4228'],
          subArcs: [{ limit: 20 }, { limit: 40 }, {}, {}, {}],
          padding: 0.02,
          width: 0.3,
        }}
        pointer={{
          elastic: true,
          animationDelay: 0,
        }}
      />

      {/* Display soil status */}
      <h4 className="white-h4">Soil Status: {soilStatus}</h4>

      {/* Display pump state */}
      <h4 className="white-h4">Pump is {pumpState}</h4>

      {/* Button to toggle pump state */}
      <center>
        <button
          type="button"
          className="btn btn-outline-success"
          onClick={togglePump}
          disabled={isToggling} // Disable button while toggling
        >
          {isToggling ? 'Toggling...' : 'Toggle Pump'}
        </button>
      </center>
    </div>
  );
}

export default Gauge2;
