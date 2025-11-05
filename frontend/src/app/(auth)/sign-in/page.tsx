"use client";

import auth from "../auth.module.scss";

export default function Page() {
  return (
    <div className={auth["sign-in"]}>
      <form>
        <h1>Welcome back!</h1>
        <label>Email</label>
        <input type="email" placeholder="Enter your email address..." />
        <label>Password</label>
        <input type="password" placeholder="Enter your password..." />
        <input
          type="submit"
          value="Sign in"
          className={auth["sign-in-button"]}
        ></input>
      </form>
      <button className={auth["create-account"]}>Create an account</button>
      {/* TODO: figure out how to apply style to id */}
      <p className={auth["reset-password"]}>Reset password</p>
    </div>
  );
}
