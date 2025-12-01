"use client";

import { useState } from "react";
import auth from "../auth.module.scss";
import { auth as firebaseAuth } from "../../../config/firebase-config";
import { useSignInWithEmailAndPassword } from "react-firebase-hooks/auth";
import { useRouter } from "next/navigation";
import { saveAuthTokenInCookies } from "../../../shared/utils/save-auth-token-in-cookies.util";

export default function Page() {
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [signInWithEmailAndPassword] =
    useSignInWithEmailAndPassword(firebaseAuth);

  const onSubmit = async () => {
    if (!email || !password) {
      alert("Please enter your email and password.");
      return;
    }

    const result = await signInWithEmailAndPassword(email, password);

    if (!result) {
      alert("Failed to sign in. Check your credentials.");
      return;
    }

    const user = firebaseAuth.currentUser;

    if (user) {
      const token = await user.getIdToken();

      if (token) {
        const res = await saveAuthTokenInCookies(token);
        console.log(
          res.ok
            ? "Auth token saved successfully"
            : "Failed to save auth token, status: " + res.status
        );
      }
    } else {
      console.error("Firebase auth user is null after sign in.");
      alert("Something went wrong signing you in.");
      return;
    }

    router.push("/dashboard");
  };

  return (
    <div className={auth["auth-form"]}>
      <h1>Welcome back!</h1>

      <label>Email</label>
      <input
        type="email"
        placeholder="Enter your email address..."
        onChange={(e) => setEmail(e.target.value)}
      />

      <label>Password</label>
      <input
        type="password"
        placeholder="Enter your password..."
        onChange={(e) => setPassword(e.target.value)}
      />

      <input
        type="submit"
        value="Sign in"
        onClick={onSubmit}
        className={auth["submit-button"]}
      />

      <button className={auth["redirect-button"]}>Create an account</button>

      <p className={auth["reset-password"]}>Reset password</p>
    </div>
  );
}
