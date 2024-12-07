import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route, useNavigate } from "react-router-dom";
import InputField from "./components/InputField";

const App = () => {
  const [isSignUp, setIsSignUp] = useState(false);
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
  });
  const navigate = useNavigate();  // Hook for navigation

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const endpoint = isSignUp ? "signup" : "login";
    const response = await fetch(`http://localhost:5000/${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    });

    const data = await response.json();

    if (response.ok) {
      console.log(data.message);
      // On successful login/signup, navigate to the next page
      navigate("/dashboard"); // Replace "/dashboard" with your desired route
    } else {
      console.error(data.error);
      // Handle error (show alert, error message, etc.)
    }

    // Reset form after submission (optional)
    setFormData({ name: "", email: "", password: "" });
  };

  const toggleForm = () => {
    setIsSignUp(!isSignUp);
  };

  return (
    <div style={{ backgroundImage: "url('background.png')" }}>
      <div className="login-container">
        <h2 className="form-title">{isSignUp ? "Sign Up" : "Log In"}</h2>
        <form onSubmit={handleSubmit} className="login-form">
          {isSignUp && (
            <InputField
              type="text"
              placeholder="Name"
              icon="person"
              name="name"
              value={formData.name}
              onChange={handleChange}
            />
          )}
          <InputField
            type="email"
            placeholder="Email address"
            icon="email"
            name="email"
            value={formData.email}
            onChange={handleChange}
          />
          <InputField
            type="password"
            placeholder="Password"
            icon="password"
            name="password"
            value={formData.password}
            onChange={handleChange}
          />
          <button type="submit" className="login-button">
            {isSignUp ? "Sign Up" : "Log In"}
          </button>
        </form>

        <p className="signup-prompt">
          {isSignUp ? (
            <>
              Already have an account?{" "}
              <a href="#" className="signup-link" onClick={toggleForm}>
                Log in
              </a>
            </>
          ) : (
            <>
              Don&apos;t have an account?{" "}
              <a href="#" className="signup-link" onClick={toggleForm}>
                Sign up
              </a>
            </>
          )}
        </p>
      </div>
    </div>
  );
};

const Dashboard = () => {
  return <h2>Welcome to the Dashboard</h2>;  // This is the page after login/signup
};

const AppWrapper = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </Router>
  );
};

export default AppWrapper;
  