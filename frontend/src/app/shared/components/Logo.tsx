"use client";

import { useRouter } from "next/navigation";
import Image from "next/image";

export function HorizontalLogo({
  height = 32,
  width = height * 3.859375,
}: {
  height?: number;
  width?: number;
}) {
  const router = useRouter();

  return (
    <Image
      src="/horizontal-logo.svg"
      alt="Logotype Icon"
      height={height}
      width={width}
      style={{ objectFit: "contain", cursor: "pointer" }}
      onClick={() => {
        router.push("/");
      }}
    />
  );
}
