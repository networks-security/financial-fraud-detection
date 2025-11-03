"use client";

import shared from "../shared.module.scss";

export function SignInButton() {
  return <button className={shared["sign-in"]}>Sign In</button>;
}

export function SignUpButton() {
  return <button className={shared["sign-up"]}>Sign Up</button>;
}
