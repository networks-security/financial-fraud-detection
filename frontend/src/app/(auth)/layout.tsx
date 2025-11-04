import { ReactNode } from "react";
import Header from "../shared/components/Header";
import auth from "./auth.module.scss";
import Image from "next/image";

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <>
      <Header />
      <div className={auth.layout}>
        <div className={auth["left-panel"]}>
          <p className="bold-text">Smarter eyes against fraud</p>
          <p className={auth["left-panel-description"]}>
            Designed for accuracy, speed, and trust. Stay one step ahead of
            fraud.
          </p>
          <div className={auth["bank-image"]}>
            <Image
              src="/bank.svg"
              fill
              style={{
                objectFit: "contain",
                maxWidth: "1000px",
              }}
              alt="Bank Image"
            ></Image>
          </div>
        </div>
        <div>{children}</div>
      </div>
    </>
  );
}
