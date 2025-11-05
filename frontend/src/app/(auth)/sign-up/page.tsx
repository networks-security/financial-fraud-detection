import auth from "../auth.module.scss";

export default function Page() {
  return (
    <div className={auth["auth-form"]}>
      <form>
        <h1>Welcome back!</h1>
        <label>Email</label>
        <input type="email" placeholder="Enter your email address..." />
        <label>Create password</label>
        <input type="password" placeholder="Enter your password..." />
        <label>Confirm password</label>
        <input type="password" placeholder="Repeat your password..." />
        <input
          type="submit"
          value="Sign up"
          className={auth["submit-button"]}
        ></input>
      </form>
      <button className={auth["redirect-button"]}>Sign in instead</button>
    </div>
  );
}
