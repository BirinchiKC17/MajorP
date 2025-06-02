import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import Login from "./login";

function Home() {
  return (
    <div
      style={{
        height: "100vh",
        fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
        background: "linear-gradient(135deg, #667eea, #764ba2)",
        color: "white",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        flexDirection: "column",
        padding: "1rem",
        textAlign: "center",
      }}
    >
      <h1>Welcome to the React Vite App</h1>
      <p>
        Navigate to{" "}
        <a
          href="/login"
          style={{ color: "#cbd5e1", textDecoration: "underline" }}
        >
          Login
        </a>{" "}
        to get started.
      </p>
    </div>
  );
}

export default function AppRoutes() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}
