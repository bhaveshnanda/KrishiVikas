import React from 'react';
import Gauge1 from './Gauge1.jsx'
import Gauge2 from './Gauge2.jsx'
import Gauge3 from './Gauge3.jsx'
import Thermometergauge from './Thermometer.jsx'

function Dashboard() {

  return (
<main>
        <div className="thermo-card">
          <h2>Temperature</h2>
          <br></br>
          <div className="thermo-chart text-center">
            <Thermometergauge />
          </div>
        </div>

        <div className="card">
          <center><h2>Humidity</h2></center>
          
          <div className="chart">
            <Gauge1 />
          </div>
        </div>
        <div className="card">
        <center><h2>Soil Moisture</h2></center>
          
          <div className="chart">
            <Gauge2 />
          </div>
        </div>
        <div className="card">
        <center><h2>Air Quality</h2></center>
          
          <div className="chart">
            <Gauge3 />
          </div>
        </div>
      </main>
  )
}

export default Dashboard
