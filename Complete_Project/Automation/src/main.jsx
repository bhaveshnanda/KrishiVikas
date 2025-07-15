import React from 'react'
import ReactDOM from 'react-dom/client'
import './index.css'
import Navbar from './landing_page/Navbar.jsx'
import Dashboard from './landing_page/Dashboard.jsx'
import Footer from './landing_page/Footer.jsx'
// import ReactDOM from "react-dom";
// import { StrictMode } from "react";

import App from "./landing_page/App.jsx";


ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <Navbar />
    <Dashboard />
    <App/>
    <Footer />
  </React.StrictMode>,
)
