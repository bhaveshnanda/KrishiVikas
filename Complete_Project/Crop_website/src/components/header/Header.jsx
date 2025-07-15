import React from "react";
import { Link } from "react-router-dom";
import "./Header.css";
import logo from "./main-logo.png";
const Header = () => {
  return (
    <div className="header">
      <div className="logo_container">
        <div className="logo1">
          <img src={logo} alt="logo" />
        </div>
      </div>
      <ul>
        <li className="crop-header-option">
          <Link to="/crop">
            Crop & Fertilizer <div>Recommendation</div>
          </Link>
        </li>
        {/* <li className="fertilizer-header-option">
          <Link to="/fertilizer"> Fertilizer Recommendation </Link>
        </li> */}
        <li className="project-header-option">
          <Link
            to="#"
            onClick={() => {
              window.location.href = "http://localhost:4000/";
            }}
          >
            Crop Prediction
          </Link>
        </li>
        <li>
          <a href="http://localhost:5000/">
            <button className="login_1">
              <i className="fa fa-fw fa-user"></i>Home
            </button>
          </a>
        </li>
      </ul>
    </div>
  );
};

export default Header;
