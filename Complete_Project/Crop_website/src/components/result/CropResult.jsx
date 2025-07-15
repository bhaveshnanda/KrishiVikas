import React, { useEffect } from "react";
import Header from "../header/Header";
import "./CropResult.css";
import { useNavigate, useLocation } from "react-router-dom";
import { output_descriptions, label_image_paths } from "../crop/CropOutputs";

export function CropResult() {
  const navigate = useNavigate();
  const location = useLocation();
  const locationState = location.state;

  // Debugging logs
  console.log("LOCATION: ", location);
  console.log("LOCATION STATE: ", locationState);

  useEffect(() => {
    if (!locationState || !locationState.predicted_crop) {
      console.log("Redirecting to /crop because locationState is missing...");
      navigate("/crop");
    }
  }, [locationState, navigate]);

  if (!locationState || !locationState.predicted_crop) {
    console.log("LocationState is null or predicted_crop is missing");
    return null;
  }

  const predicted_crop = locationState.predicted_crop;
  const recommended_fertilizer =
    locationState.recommended_fertilizer || "No fertilizer recommendation available";

  // 🛠️ Extract the crop label from the array
  const cropName0 = predicted_crop[0]?.name || "UNKNOWN CROP";
  const cropLabel0 = predicted_crop[0]?.preference || "UNKNOWN CROP";

  // 🛠️ Extract the crop label from the array
  const cropName1 = predicted_crop[1]?.name || "UNKNOWN CROP";
  const cropLabel1 = predicted_crop[1]?.preference || "UNKNOWN CROP";



  // 🖼️ Use label to get image and description
  const output_image_path0 = label_image_paths[cropName0] || "";
  const output_description0 = output_descriptions[cropName0] || "No description available";

  // 🖼️ Use label to get image and description
  const output_image_path1 = label_image_paths[cropName1] || "";
  const output_description1 = output_descriptions[cropName1] || "No description available";



  // Debug logs
  console.log("Predicted Crop Array:", predicted_crop);
  console.log("Selected Crop Label:", cropName0);
  console.log("Recommended Fertilizer:", recommended_fertilizer);
  console.log("Image Path:", output_image_path0);

  return (
    <>
      <Header />
      {recommended_fertilizer && (
        <div className="fertilizer-result-container">
        <div className="fertilizer-result-card">
          <p className="fertilizer-title">Recommended Fertilizer:</p>
          <div className="fertilizer-layout">
            {recommended_fertilizer.trim().split(" ").length > 1 && (
              <div className="label-row">
                <span className="label">N</span>
                <span className="label">P</span>
                <span className="label">K</span>
              </div>
            )}
            <div className="value-row">
              {recommended_fertilizer.trim().split(" ").map((value, index, arr) => (
                <React.Fragment key={index}>
                  <span className="value">{value}</span>
                  {index < arr.length - 1 && <div className="separator" />}
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>
        </div>
      )}

      <div className="crop-result-row">
  {/* First Crop */}
  <div className="crop-result-card">
    <p className="crop-result-p">
      You should grow <b>{cropName0.toUpperCase()}</b> in your farm!
      <br />It Should be your {cropLabel0}
    </p>
    {output_image_path0 && (
      <img
        className="crop-result-img"
        src={output_image_path0}
        alt={cropName0}
      />
    )}
    <p className="crop-result-description"> {output_description1} </p>
  </div>

  {/* Second Crop */}
  <div className="crop-result-card">
    <p className="crop-result-p">
      You should grow <b>{cropName1.toUpperCase()}</b> in your farm!
      <br />It Should be your {cropLabel1}
    </p>
    {output_image_path1 && (
      <img
        className="crop-result-img"
        src={output_image_path1}
        alt={cropName1}
      />
    )}
    <p className="crop-result-description"> {output_description1} </p>
  </div>
</div>
<br></br><br></br>
      <button className="crop-try-btn" onClick={() => navigate("/crop")}>
        Try again?
      </button>
    </>
  );
}
   