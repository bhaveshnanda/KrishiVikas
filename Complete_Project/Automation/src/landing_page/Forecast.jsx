import React, { useEffect, useState } from "react";
import axios from "axios";
import ReactAnimatedWeather from "react-animated-weather";

function Forecast({ weather }) {
  const { data } = weather;
  const [rawHourlyData, setRawHourlyData] = useState([]);
  const [processedForecast, setProcessedForecast] = useState([]);
  const [isCelsius, setIsCelsius] = useState(true);

  // Fetch the hourly forecast using the Weatherbit API.
  useEffect(() => {
    const fetchForecastData = async () => {
      if (!data.coord) return; // Ensure we have coordinates
      const { lat, lon } = data.coord;
      const apiKey = "68a7b76b2d6e4d7e815620ff96110a13";
      // Weatherbit's hourly forecast endpoint (fetching 16 hours)
      const url = `https://api.weatherbit.io/v2.0/forecast/hourly?lat=${lat}&lon=${lon}&key=${apiKey}&hours=16`;

      try {
        const response = await axios.get(url);
        // Map Weatherbit's response data to match our expected format.
        // Weatherbit returns an array in response.data.data.
        const mappedData = response.data.data.map((item) => ({
          // Convert Weatherbit's timestamp_local (a string) into Unix timestamp (in seconds)
          dt: Math.floor(new Date(item.timestamp_local).getTime() / 1000),
          temp: item.temp,
          // Wrap the weather object in an array to maintain compatibility with existing code.
          weather: [item.weather],
        }));
        setRawHourlyData(mappedData);
      } catch (error) {
        console.error(
          "Error fetching hourly forecast data from Weatherbit:",
          error
        );
      }
    };

    fetchForecastData();
  }, [data]);

  // Process the raw hourly data to create a forecast array with half‑hour intervals.
  useEffect(() => {
    if (rawHourlyData.length < 2) return; // Need at least two hourly entries

    const processed = [];
    // Loop through hourly entries and interpolate a half‑hour entry between each pair.
    for (let i = 0; i < rawHourlyData.length - 1; i++) {
      const current = rawHourlyData[i];
      const next = rawHourlyData[i + 1];

      // Push the current full-hour forecast.
      processed.push({
        dt: current.dt,
        temp: current.temp,
        weather: current.weather, // use the array from current hour
      });

      // Create a half‑hour forecast by linearly interpolating the temperature.
      // (For simplicity, we use the current hour’s weather icon and description.)
      processed.push({
        dt: current.dt + 1800, // 1800 seconds = 30 minutes later
        temp: (current.temp + next.temp) / 2,
        weather: current.weather,
      });
    }
    // Optionally, include the final hourly forecast.
    processed.push(rawHourlyData[rawHourlyData.length - 1]);

    setProcessedForecast(processed);
  }, [rawHourlyData]);

  // Format the Unix timestamp (in seconds) into a time string like "14:00".
  const formatTime = (timestamp) => {
    const date = new Date(timestamp * 1000);
    let hours = date.getHours();
    let minutes = date.getMinutes();
    minutes = minutes < 10 ? `0${minutes}` : minutes;
    return `${hours}:${minutes}`;
  };

  const toggleTemperatureUnit = () => {
    setIsCelsius((prevState) => !prevState);
  };

  const convertToFahrenheit = (temp) => Math.round((temp * 9) / 5 + 32);

  const renderTemperature = (temp) =>
    isCelsius ? Math.round(temp) : convertToFahrenheit(temp);

  // Build the current weather icon URL.
  const currentIconUrl =
    data.weather && data.weather[0]
      ? `http://openweathermap.org/img/wn/${data.weather[0].icon}@2x.png`
      : "";

  return (
    <div>
      {/* Current Weather */}
      <div className="city-name">
        <h2>
          {data.name}, <span>{data.sys && data.sys.country}</span>
        </h2>
      </div>
      <div className="temp">
        {currentIconUrl && (
          <img
            src={currentIconUrl}
            alt={data.weather && data.weather[0].description}
            className="temp-icon"
          />
        )}
        {renderTemperature(data.main && data.main.temp)}
        <sup className="temp-deg" onClick={toggleTemperatureUnit}>
          {isCelsius ? "°C" : "°F"} | {isCelsius ? "°F" : "°C"}
        </sup>
      </div>
      <p className="weather-des">
        {data.weather && data.weather[0].description}
      </p>
      <div className="weather-info">
        <div className="col">
          <ReactAnimatedWeather icon="WIND" size="40" />
          <div>
            <p className="wind">{data.wind && data.wind.speed} m/s</p>
            <p>Wind speed</p>
          </div>
        </div>
        <div className="col">
          <ReactAnimatedWeather icon="RAIN" size="40" />
          <div>
            <p className="humidity">{data.main && data.main.humidity}%</p>
            <p>Humidity</p>
          </div>
        </div>
      </div>

      {/* Forecast Section */}
      <div className="forecast">
        <h3>Forecast ( Intervals):</h3>
        <div className="forecast-container">
          {processedForecast.slice(0, 10).map((entry) => (
            <div className="forecast-entry" key={entry.dt}>
              <p className="forecast-time">{formatTime(entry.dt)}</p>
              {entry.weather && entry.weather[0] && (
                <img
                  className="forecast-icon"
                  src={`https://www.weatherbit.io/static/img/icons/${entry.weather[0].icon}.png`}
                  alt={entry.weather[0].description}
                />
              )}
              <p className="forecast-temp">{renderTemperature(entry.temp)}°</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default Forecast;
