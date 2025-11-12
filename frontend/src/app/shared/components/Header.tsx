import shared from "../shared.module.scss";
import { HorizontalLogo } from "./Logo";
import { SignUpButton, SignInButton } from "./HeaderButtons";

import Link from "next/link";

export default function Header() {
  return (
    <div>
      <div
        className={shared.header}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          position: "fixed",
        }}
      >
        {/* Left Header Side */}
        <span className={shared.left}>
          <HorizontalLogo />
        </span>

        {/* Right Header Side */}
        <span className={shared.right}>
          <Link href={"/dashboard"}>
            <p>Dashboard</p>
          </Link>
          {/* // TODO: once auth is implemented, configure so that fake data is provided to the client, i.g. guest account */}
          <Link href={"/dashboard"}>
            <p>Try it out</p>
          </Link>
          <SignInButton />
          <SignUpButton />
        </span>
      </div>

      {/* Ensure content stays below the Header */}
      <div style={{ height: "var(--header-height)" }}></div>
    </div>
  );
}
