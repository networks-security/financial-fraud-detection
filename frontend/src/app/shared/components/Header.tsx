import shared from "../shared.module.scss";
import { HorizontalLogo } from "./Logo";
import { SignUpButton, SignInButton } from "./HeaderButtons";

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
          <p>Dashboard</p>
          <p>Try it out</p>
          <SignInButton />
          <SignUpButton />
        </span>
      </div>

      {/* Ensure content stays below the Header */}
      <div style={{ height: "var(--header-height)" }}></div>
    </div>
  );
}
