import React, { useEffect, useState } from 'react';
import Header from "../header/Header.jsx";
import "./CropPage.css";
import { TextField, FormControlLabel, Switch, Button } from "@mui/material";
import { useNavigate } from "react-router-dom";
import LinearProgress from "@mui/material/LinearProgress";

// Min-Max values for validation
export const crop_value_ranges = {
  nitrogen: [5, 150],
  phosphorous: [5, 145],
  potassium: [5, 205],
  temperature: [5, 50],
  humidity: [5, 100],
  ph: [3, 10],
  rainfall: [20, 300],
};

const CROP_ENDPOINT = "http://127.0.0.1:7050/crop_recommend";
const SENSOR_ENDPOINT = "http://192.168.43.2452readSensor"; // ESP32 endpoint

export function CropPage() {
  const navigate = useNavigate();

  const [alertVisible, setAlertVisible] = useState(false);
  const [alertMessage, setAlertMessage] = useState('');
  const [autoMode, setAutoMode] = useState(false);

  const [inputs, setInputs] = useState({
    nitrogen: '',
    phosphorous: '',
    potassium: '',
    temperature: '',
    humidity: '',
    ph: '',
    rainfall: '',
  });

  // Fetch data from ESP32 when autoMode is enabled
  useEffect(() => {
    if (autoMode) {
      fetch(SENSOR_ENDPOINT)
        .then(res => res.json())
        .then(data => {
          setInputs(prev => ({
            ...prev,
            temperature: data.temperature || '',
            humidity: data.humidity || '',
          }));
        })
        .catch(err => {
          console.error("Sensor fetch failed:", err);
        });
    }
  }, [autoMode]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setInputs(prev => ({ ...prev, [name]: value }));
  };

  const validateInputs = () => {
    let allInvalid = true; // assume all are invalid
  
    for (const [key, value] of Object.entries(inputs)) {
      if (value === '') {
        setAlertMessage(`Please enter a value for ${key}.`);
        setAlertVisible(true);
        return false;
      }
  
      const [min, max] = crop_value_ranges[key] || [0, Infinity];
      const numValue = parseFloat(value);
  
      if (numValue >= min && numValue <= max) {
        allInvalid = false; // found at least one valid input
      }
    }
  
    if (allInvalid) {
      setAlertMessage("No crops are suitable for this type of soil and environment.");
      setAlertVisible(true);
      window.alert("❌ No crops are suitable for this type of soil and environment.");
      return false;
    }
  
    return true;
  };
  

  const handleClick = () => {
    if (!validateInputs()) return;

    const progressBar = document.querySelector(".crop-progress-bar");
    progressBar.style.display = "block";
    progressBar.style.visibility = "visible";

    const data = {
      array: [
        parseFloat(inputs.nitrogen),
        parseFloat(inputs.phosphorous),
        parseFloat(inputs.potassium),
        parseFloat(inputs.temperature),
        parseFloat(inputs.humidity),
        parseFloat(inputs.ph),
        parseFloat(inputs.rainfall),
      ],
    };

    fetch(CROP_ENDPOINT, {
      method: "POST",
      body: JSON.stringify(data),
      headers: { "Content-Type": "application/json" },
    })
      .then((response) => {
        if (!response.ok) throw new Error("Network error");
        return response.json();
      })
      .then((data) => {
        navigate("/crop_result", {
          state: {
            predicted_crop: data.recommended_crops,
            recommended_fertilizer: data.recommended_fertilizer,
          },
        });
      })
      .catch((error) => {
        console.error("Fetch error:", error);
        setAlertMessage("Something went wrong. Please try again.");
        setAlertVisible(true);
      })
      .finally(() => {
        progressBar.style.display = "none";
      });
  };

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === "Enter") handleClick();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [inputs]);

  return (
    <>
      <Header />
      <LinearProgress
        style={{ visibility: "hidden", display: "none" }}
        className="crop-progress-bar"
        color="success"
      />
      <p className="crop-p">
        Enter soil characteristics to find the best <b>Crop & Fertilizer Recommendation</b> for your farm.
      </p>
      <div className="crop-container">
        <FormControlLabel
          control={
            <Switch
              checked={autoMode}
              onChange={() => setAutoMode(!autoMode)}
              color="primary"
            />
          }
          label="Automatic Sensor Input"
        />

        <TextField
          name="nitrogen"
          label="Ratio of Nitrogen"
          variant="outlined"
          color="success"
          type="number"
          value={inputs.nitrogen}
          onChange={handleInputChange}
        />
        <TextField
          name="phosphorous"
          label="Ratio of Phosphorous"
          variant="outlined"
          color="success"
          type="number"
          value={inputs.phosphorous}
          onChange={handleInputChange}
        />
        <TextField
          name="potassium"
          label="Ratio of Potassium"
          variant="outlined"
          color="success"
          type="number"
          value={inputs.potassium}
          onChange={handleInputChange}
        />
        <TextField
          name="temperature"
          label="Temperature (°C)"
          variant="outlined"
          color="success"
          type="number"
          value={inputs.temperature}
          onChange={handleInputChange}
          disabled={autoMode}
        />
        <TextField
          name="humidity"
          label="Humidity (%)"
          variant="outlined"
          color="success"
          type="number"
          value={inputs.humidity}
          onChange={handleInputChange}
          disabled={autoMode}
        />
        <TextField
          name="ph"
          label="pH Level"
          variant="outlined"
          color="success"
          type="number"
          value={inputs.ph}
          onChange={handleInputChange}
        />
        <TextField
          name="rainfall"
          label="Rainfall (mm)"
          variant="outlined"
          color="success"
          type="number"
          value={inputs.rainfall}
          onChange={handleInputChange}
        />

        <Button
          variant="contained"
          color="success"
          onClick={handleClick}
          style={{ marginTop: '1rem' }}
        >
          Predict Crops & Fertilizer
        </Button>

        {alertVisible && (
          <p className="alert-message" style={{ color: "red", marginTop: "10px" }}>
            {alertMessage}
          </p>
        )}
      </div>
    </>
  );
}
