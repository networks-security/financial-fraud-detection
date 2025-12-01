"use client";
import { useState } from "react";
import auth from "../auth.module.scss";
import { auth as firebaseAuth } from "../../../config/firebase-config";
import { useCreateUserWithEmailAndPassword } from "react-firebase-hooks/auth";
import { sendEmailVerification } from "firebase/auth";
import { useRouter } from "next/navigation";
import { saveAuthTokenInCookies } from "../../../shared/utils/save-auth-token-in-cookies.util";

export default function Page() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [createUser] = useCreateUserWithEmailAndPassword(firebaseAuth);

  const onSubmit = async () => {
    const password1 = document.getElementById("password1") as HTMLInputElement;
    const password2 = document.getElementById("password2") as HTMLInputElement;

    if (password1.value != password2.value) {
      alert("Passwords do not match!");
      return;
    }

    console.log(await createUser(email, password));
    const user = firebaseAuth.currentUser;
    if (user) {
      const token = await firebaseAuth.currentUser?.getIdToken();

      if (token) {
        const res = await saveAuthTokenInCookies(token);
        console.log(
          res.ok
            ? "Auth token saved successfully"
            : "Failed to save auth token, status: " + res.status
        );
      }

      await sendEmailVerification(user);
    } else {
      console.error(
        "Could not send a verification email because current user is null"
      );
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
        onChange={(e) => {
          setEmail(e.target.value);
        }}
      />
      <label>Create password</label>
      <input
        id="password1"
        type="password"
        placeholder="Enter your password..."
        onChange={(e) => {
          setPassword(e.target.value);
        }}
      />
      <label>Confirm password</label>
      <input
        id="password2"
        type="password"
        placeholder="Repeat your password..."
        onChange={(e) => {
          setPassword(e.target.value);
        }}
      />
      <input
        type="submit"
        value="Sign up"
        onClick={onSubmit}
        className={auth["submit-button"]}
      ></input>
      <button className={auth["redirect-button"]}>Sign in instead</button>
    </div>
  );
}
