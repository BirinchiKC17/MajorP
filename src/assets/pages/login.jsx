import { useState } from "react";

const Login = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
  });
  const [message, setMessage] = useState("");

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (isLogin) {
      if (!formData.email || !formData.password) {
        setMessage("Please enter email and password.");
        return;
      }
      setMessage(`Logging in with email: ${formData.email}`);
    } else {
      if (!formData.name || !formData.email || !formData.password) {
        setMessage("Please fill all fields.");
        return;
      }
      setMessage(`Registering user: ${formData.name}`);
    }
  };

  return (
    <div
      style={{
        maxWidth: 320,
        margin: "100px auto",
        padding: 20,
        border: "1px solid #ccc",
        borderRadius: 8,
      }}
    >
      <h2 style={{ textAlign: "center" }}>{isLogin ? "Login" : "Register"}</h2>
      <form onSubmit={handleSubmit}>
        {!isLogin && (
          <div style={{ marginBottom: 12 }}>
            <label>Name</label>
            <br />
            <input
              name="name"
              value={formData.name}
              onChange={handleChange}
              style={{ width: "100%", padding: 8, boxSizing: "border-box" }}
              required={!isLogin}
            />
          </div>
        )}

        <div style={{ marginBottom: 12 }}>
          <label>Email</label>
          <br />
          <input
            type="email"
            name="email"
            value={formData.email}
            onChange={handleChange}
            style={{ width: "100%", padding: 8, boxSizing: "border-box" }}
            required
          />
        </div>

        <div style={{ marginBottom: 12 }}>
          <label>Password</label>
          <br />
          <input
            type="password"
            name="password"
            value={formData.password}
            onChange={handleChange}
            style={{ width: "100%", padding: 8, boxSizing: "border-box" }}
            required
          />
        </div>

        <button
          type="submit"
          style={{
            width: "100%",
            padding: 10,
            backgroundColor: "#007BFF",
            color: "white",
            border: "none",
            borderRadius: 4,
          }}
        >
          {isLogin ? "Login" : "Register"}
        </button>
      </form>

      <p style={{ textAlign: "center", marginTop: 12 }}>
        {isLogin ? "Don't have an account?" : "Already have an account?"}{" "}
        <button
          onClick={() => {
            setIsLogin(!isLogin);
            setFormData({ name: "", email: "", password: "" });
            setMessage("");
          }}
          style={{
            background: "none",
            border: "none",
            color: "#007BFF",
            cursor: "pointer",
            textDecoration: "underline",
            padding: 0,
            fontSize: "inherit",
          }}
        >
          {isLogin ? "Register here" : "Login here"}
        </button>
      </p>

      {message && (
        <p style={{ textAlign: "center", marginTop: 12, color: "green" }}>
          {message}
        </p>
      )}
    </div>
  );
};

export default Login;
