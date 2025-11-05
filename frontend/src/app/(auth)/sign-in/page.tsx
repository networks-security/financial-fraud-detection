"use client";

import auth from "../auth.module.scss";

export default function Page() {
  return (
    <div className={auth["auth-form"]}>
      <form>
        <h1>Welcome back!</h1>
        <label>Email</label>
        <input type="email" placeholder="Enter your email address..." />
        <label>Password</label>
        <input type="password" placeholder="Enter your password..." />
        <input
          type="submit"
          value="Sign in"
          className={auth["submit-button"]}
        ></input>
      </form>
      <button className={auth["redirect-button"]}>Create an account</button>
      <p className={auth["reset-password"]}>Reset password</p>
    </div>
  );
}
