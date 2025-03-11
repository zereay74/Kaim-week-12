import React, { useEffect, useState } from "react";
import axios from "axios";
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from "recharts";

const Predictions = () => {
  const [predictions, setPredictions] = useState([]);
  const [dateRange, setDateRange] = useState("all");

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/predict")
      .then(response => {
        // Format Date to "YYYY-MM-DD" for easier viewing
        const formattedData = response.data.map(item => ({
          ...item,
          Date: new Date(item.Date).toLocaleDateString()
        }));
        setPredictions(formattedData);
      })
      .catch(error => {
        console.error("Error fetching predictions:", error);
      });
  }, []);

  // Function to filter predictions based on selected date range
  const getFilteredPredictions = () => {
    if (dateRange === "all") {
      return predictions;
    }
    const now = new Date();
    let threshold;
    if (dateRange === "last7days") {
      threshold = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    } else if (dateRange === "last30days") {
      threshold = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
    }
    return predictions.filter(pred => {
      // Parse the formatted date back into a Date object
      const predDate = new Date(pred.Date);
      return predDate >= threshold;
    });
  };

  const filteredPredictions = getFilteredPredictions();

  return (
    <div>
      <h2>Sales Predictions</h2>
      
      <div style={{ marginBottom: "20px" }}>
        <label htmlFor="dateRange">Select Date Range: </label>
        <select 
          id="dateRange" 
          value={dateRange} 
          onChange={(e) => setDateRange(e.target.value)}
        >
          <option value="all">All Data</option>
          <option value="last7days">Last 7 Days</option>
          <option value="last30days">Last 30 Days</option>
        </select>
      </div>
      
      {/* Table for displaying predictions */}
      <table border="1" style={{ width: "100%", marginBottom: "20px" }}>
        <thead>
          <tr>
            <th>Date</th>
            <th>ML Model Prediction</th>
            <th>DL Model Prediction</th>
          </tr>
        </thead>
        <tbody>
          {filteredPredictions.map((pred, index) => (
            <tr key={index}>
              <td>{pred.Date}</td>
              <td>{pred.ML_Model_Predictions}</td>
              <td>{pred.DL_Model_Predictions}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Combined line chart for visualizing predictions */}
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={filteredPredictions} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="Date" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip />
          <Legend verticalAlign="top" height={36} />
          <Line type="monotone" dataKey="ML_Model_Predictions" stroke="#8884d8" activeDot={{ r: 8 }} />
          <Line type="monotone" dataKey="DL_Model_Predictions" stroke="#82ca9d" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default Predictions;
